"""

This is my final QAT script.

The goal is to use the code from both examples together, in order to export an .xmodel file
of the model you care about.

The real challenge is installing enough dependencies that you can load your model to pass 
to the QAT processor.

"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from pytorch_nndct.apis import torch_quantizer, dump_xmodel, QatProcessor
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))
    


if __name__ == '__main__':
#This first section is from the U-Net script, in order to load a .pth file

    opt = TrainOptions().parse()
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)


    transform_list = [transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    transform = transforms.Compose(transform_list)

    calibration_images = []
    for i, data in enumerate(dataset):
        if(i > 10):
            break
        img = np.array(Image.open(data['A_paths'][0]).convert('RGB'))
        img = transform(img[:, 0:256, :])
        calibration_images.append(img)
        

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

#This section is from the qat processor script

    input = torch.randn([opt.batch_size, 3, 256, 256])

    device = torch.device('cpu')
    model.load_networks("latest")

    qat_processor = QatProcessor(model.netG, input, 8, device=device)

    quantized_model = qat_processor.trainable_model()
    model.netG = quantized_model
    model.set_optimizers(opt)

    #normal train

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            print(i)
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
                
                # if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                #     save_result = total_iters % opt.update_html_freq == 0
                #     model.compute_visuals()
                #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                # if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                #     losses = model.get_current_losses()
                #     t_comp = (time.time() - iter_start_time) / opt.batch_size
                #     visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                #     if opt.display_id > 0:
                #         visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses) 


            #This is from when I loading a fully trained model, and was saving it every 5 epochs to
            #hand pick the best model.
            
            if(i % 5 == 0):
                deployable_model = qat_processor.convert_to_deployable(quantized_model, "qat_results")

                #this middle bit prints evaulation images to see which .pth is best
                for j, calib_image in enumerate(calibration_images):
                    if(j > 6):
                        break
                    # save_img(calib_image, opt.dataroot+"/calib_images/input_" + str(epoch) + "_" +str(i) + "_" + str(j) + ".png")
                    input = calib_image.unsqueeze(0).to(device)
                    out = deployable_model(input)
                    out_img = out.detach().squeeze(0).cpu()
                    save_img(out_img, opt.dataroot+"/calib_images/output_" + str(epoch) + "_" +str(i) + "_" + str(j)  + ".png")
            
                torch.save(quantized_model.state_dict(), "./quant_checkpoints/" +opt.name + "/QAT_quantized_model_epoch_" + str(epoch) + "_" + str(i) + ".pth")
            

        #     if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
        #         print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
        #         save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
        #         model.save_networks(save_suffix)

        #     iter_data_time = time.time()
        # if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #     model.save_networks('latest')
        #     model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))


    #this bit creates the .xmodel file
    #you need to generate at least one image before you can generate an .xmodel
    deployable_model = qat_processor.convert_to_deployable(quantized_model, "qat_results")

    t1 = time.time()
    for i, calib_image in enumerate(calibration_images):
        if(i>3):
            break
        input = calib_image.unsqueeze(0).to(device)
        out = deployable_model(input)
        out_img = out.detach().squeeze(0).cpu()
        save_img(out_img, opt.dataroot+"/calib_images/final_output_" + str(i) + ".png")

    t2 = time.time()


    print("time taken: " + str(t2 - t1))
    

    qat_processor.export_xmodel(output_dir="quant", deploy_check=False)





