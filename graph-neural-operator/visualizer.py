from matplotlib import tri as mtri
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os 
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import h5py


def make_animation_lstm(gs_loader, pred_loader, evl_loader, rmse_var, path, name, args, 
                        skip = 2, save_anim = True, plot_variables = False,
                   traj_num = 0):
    
    if (args.var_type == 'sat'):
        ylabel = 'Saturation RMSE'
        legend_name = 'Gas saturation' 
        scale = 1.0
    elif (args.var_type == 'p'):
        ylabel = 'Pressure RRMSE'
        legend_name = 'Pressure (MPa)'
        scale = 1e6 # change to mpa

    gs = gs_loader[0]
    pred = pred_loader[0]
    evl = evl_loader[0]
    '''
    input gs is a dataloader and each entry contains attributes of many timesteps.

    '''
    print('Generating {} fields...'.format(name))
    #fig, axes = plt.subplots(1, 3, figsize=(36, 10))
    # currently only support outputting one mesh at a time
    num_steps = gs.y.shape[1] # for a single trajectory
    num_frames = num_steps // skip
    print(num_steps)
    plt.ion() 

    bb_min = gs.y[:, :].min()/scale # gas saturation
    bb_max = gs.y[:, :].max()/scale # use max and min velocity of gs dataset at the first step for both 
                                  # gs and prediction plots
    
    bb_min_evl = evl.y[:, :].min()/scale  # gas saturation
    bb_max_evl = evl.y[:, :].max()/scale  # use max and min velocity of gs dataset at the first step for both 
                                  # gs and prediction plots
    traj = traj_num                             
    if (plot_variables):
        name += '_rmse'
        fig = plt.figure(figsize=[36, 20])
        axes = [plt.subplot(2, 3, 1), 
            plt.subplot(2, 3, 2),
            plt.subplot(2, 3, 3),
            plt.subplot(2, 1, 2)] # add a subplot to show the evolution of saturation RMSE
    else:
        fig = plt.figure(figsize=[36, 10])
        axes = [plt.subplot(1, 3, 1), 
            plt.subplot(1, 3, 2),
            plt.subplot(1, 3, 3)]

    def single_ax_plot(count, ax, step):

        if ( count > 2 ):

            ax.set_xlabel('Iteration', fontsize = '20' )
            ax.set_ylabel( ylabel,  fontsize = '20')
            ax.plot(np.arange(0, step + 1), rmse_var[:step + 1], marker= 'o', c= 'k', linewidth=1)
            return False
            
        ax.cla()
        ax.set_aspect('equal')
        ax.set_axis_off()

        pos = gs.mesh_pos 

        if (count == 0):
            # ground truth
            velocity = gs.y[:, step]/scale
            title = 'Ground truth:'
        elif (count == 1):
            velocity = pred.y[:, step]/scale
            title = 'Prediction:'
        else: 
            velocity = evl.y[:, step]/scale
            title = 'Relative Error:'

        #triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
        triang = mtri.Triangulation(pos[:, 0], pos[:, 1])
        if (count <= 1):
            # absolute values
            mesh_plot = ax.tripcolor(triang, velocity, vmin= bb_min, vmax=bb_max,  shading='flat', cmap= 'viridis')
            ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
        else:
            # error: (pred - gs)/gs
            mesh_plot = ax.tripcolor(triang, velocity, vmin= bb_min_evl, vmax=bb_max_evl, shading='flat', cmap= 'viridis')
            ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
            #ax.triplot(triang, lw=0.5, color='0.5')

        ax.set_title('{} Trajectory {} Step {}'.format(title, traj, step), fontsize = '20')
        return mesh_plot
    # Frame 0
    count = 0 
    for ax in axes:
        mesh_plot = single_ax_plot(count, ax, 0)
        if ( mesh_plot == False):
            break
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
        clb.ax.tick_params(labelsize=20) 
        clb.ax.set_title( legend_name,
                        fontdict = {'fontsize': 20})
        count += 1
    # the rest of frame
    def animate(num):
        step = (num*skip) % num_steps
        count = 0
        for ax in axes:
            mesh_plot = single_ax_plot(count, ax, step)
            if ( mesh_plot == False):
                break

            cax.cla()
            clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
            clb.ax.tick_params(labelsize=20) 

            clb.ax.set_title(legend_name,
                            fontdict = {'fontsize': 20})
            count += 1

        return fig

    # Save animation for visualization
    if not os.path.exists(path):
        os.makedirs(path)
    
    if (save_anim):
        gs_anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=2000)
        writergif = animation.PillowWriter(fps=10) 
        anim_path = os.path.join(path, '{}_anim.gif'.format(name))
        gs_anim.save( anim_path, writer=writergif, dpi = 50)
        plt.show(block=True)
    else:
        pass


def make_animation_onestep(gs, pred, evl, rmse_var, path, name , args,
                           skip = 2, save_anim = True, plot_variables = False, traj_num = 0):
    '''
    input gs is a dataloader and each entry contains attributes of many timesteps.

    '''
    print('Generating {} fields...'.format(name))
    #fig, axes = plt.subplots(1, 3, figsize=(36, 10))
    if (args.var_type == 'sat'):
        ylabel = 'Saturation RMSE'
        legend_name = 'Gas saturation' 
        scale = 1.0
    elif (args.var_type == 'p'):
        ylabel = 'Pressure RRMSE'
        legend_name = 'Pressure (MPa)'
        scale = 1e6 # change to mpa

    num_steps = len(gs) # for a single trajectory
    num_frames = num_steps // skip
    print(num_steps)
    plt.ion() 

    bb_min = gs[-1].y[:, 0].min()/ scale # gas saturation
    bb_max = gs[-1].y[:, 0].max()/ scale # use max and min velocity of gs dataset at the first step for both 
                                  # gs and prediction plots
    bb_min_evl = evl[0].y[:, 0].min()/scale  # gas saturation
    bb_max_evl = evl[0].y[:, 0].max()/scale  # use max and min velocity of gs dataset at the first step for both 
                                  # gs and prediction plots
    traj = traj_num                             
    if (plot_variables):
        name += '_rmse'
        fig = plt.figure(figsize=[36, 20])
        axes = [plt.subplot(2, 3, 1), 
            plt.subplot(2, 3, 2),
            plt.subplot(2, 3, 3),
            plt.subplot(2, 1, 2)] # add a subplot to show the evolution of saturation RMSE
    else:
        fig = plt.figure(figsize=[36, 10])
        axes = [plt.subplot(1, 3, 1), 
            plt.subplot(1, 3, 2),
            plt.subplot(1, 3, 3)]

    def single_ax_plot(count, ax, step):

        if ( count > 2 ):
            ax.set_xlabel('Iteration', fontsize = '20' )
            ax.set_ylabel(ylabel,  fontsize = '20')
            ax.plot(np.arange(0, step + 1), rmse_var[:step + 1], marker= 'o', c= 'k', linewidth=1)
            return False
            
        ax.cla()
        ax.set_aspect('equal')
        ax.set_axis_off()

        pos = gs[step].mesh_pos 

        if (count == 0):
            # ground truth
            velocity = gs[step].y[:, 0]/ scale
            title = 'Ground truth:'
        elif (count == 1):
            velocity = pred[step].y[:, 0]/ scale
            title = 'Prediction:'
        else: 
            velocity = evl[step].y[:, 0]/scale
            title = 'Relative Error:'

        #triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
        triang = mtri.Triangulation(pos[:, 0], pos[:, 1])
        if (count <= 1):
            # absolute values
            mesh_plot = ax.tripcolor(triang, velocity, vmin= bb_min, vmax=bb_max,  shading='flat', cmap= 'viridis')
            ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
        else:
            # error: (pred - gs)/gs
            mesh_plot = ax.tripcolor(triang, velocity, vmin= bb_min_evl, vmax=bb_max_evl, shading='flat', cmap= 'viridis')
            ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
            #ax.triplot(triang, lw=0.5, color='0.5')

        ax.set_title('{} Trajectory {} Step {}'.format(title, traj, step), fontsize = '20')
        return mesh_plot
    # Frame 0
    count = 0 
    for ax in axes:
        mesh_plot = single_ax_plot(count, ax, 0)
        if ( mesh_plot == False):
            break
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
        clb.ax.tick_params(labelsize=20) 
        clb.ax.set_title(legend_name ,
                        fontdict = {'fontsize': 20})
        count += 1
    # the rest of frame
    def animate(num):
        step = (num*skip) % num_steps
        count = 0
        for ax in axes:
            mesh_plot = single_ax_plot(count, ax, step)
            if ( mesh_plot == False):
                break

            cax.cla()
            clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
            clb.ax.tick_params(labelsize=20) 
            clb.ax.set_title(legend_name,
                            fontdict = {'fontsize': 20})
            count += 1

        return fig

    # Save animation for visualization
    if not os.path.exists(path):
        os.makedirs(path)
    
    if (save_anim):
        gs_anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=2000)
        writergif = animation.PillowWriter(fps=10) 
        anim_path = os.path.join(path, '{}_anim.gif'.format(name))
        gs_anim.save( anim_path, writer=writergif)
        plt.show(block=True)
    else:
        pass


def visualize(loader, best_model, file_dir, args, gif_name, stats_list, comp_args, sequntial = True,
              rolling_out = False,
              make_movie = True, plot_rmse = False,
              delta_t = 0.01, skip = 1, traj_num = 0 , save_vtk = False):

    gif_name = 'Mesh{}'.format(str(traj_num)) + '_' + gif_name
    best_model.eval()
    device = args.device
    rmse_var = []
    relative_err_var = []
    viz_data = {}
    gs_data = {}
    eval_data = {}
    viz_data_loader = copy.deepcopy(loader)
    gs_data_loader = copy.deepcopy(loader)
    eval_data_loader = copy.deepcopy(loader)

    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y] = stats_list
    (mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y)=(mean_vec_x.to(device),
            std_vec_x.to(device),mean_vec_edge.to(device),std_vec_edge.to(device),mean_vec_y.to(device),std_vec_y.to(device))
    
    if (save_vtk):
        if not os.path.isdir( args.vtk_dir ):
            os.mkdir(args.vtk_dir)

        file_name = os.path.join( args.vtk_dir, 'Mesh'+ str(traj_num) +'.h5')
        data_file = h5py.File(file_name, "w")
        sg_gs, sg_pred, sg_eval = [], [], []

    for data, viz_data, gs_data, eval_data, i in zip( loader, viz_data_loader,
                                                  gs_data_loader, eval_data_loader, range(len(loader))):    
        data=data.to(args.device) 
        viz_data = viz_data.to(args.device)

        tmp_data = copy.deepcopy(viz_data)
        gs_data = gs_data.to(args.device)
        
        with torch.no_grad():
            for num in range(args.rollout_num):
                # onestep only has 1 rollout 
                # sequential model has many rollouts, but only take in one mesh at one time
                if (sequntial):
                    if (num == 0):
                        h_0 = torch.zeros(data.x.shape[0], args.hidden_dim).to(device)
                        c_0 = torch.zeros(data.x.shape[0], args.hidden_dim).to(device)

                    pred, h_0, c_0 = best_model(tmp_data, mean_vec_x,std_vec_x,
                                                mean_vec_edge,std_vec_edge, h_0, c_0)
                else:
                    # one-step predictor
                    pred = best_model(tmp_data, mean_vec_x,std_vec_x,
                                                mean_vec_edge,std_vec_edge)
                
                tmp_data.x[:, 0] = stats.unnormalize( pred.squeeze(), mean_vec_y[0], std_vec_y[0] )
                if (args.rela_perm.lower() != 'none'):
                    #print('sg mean {}'.format(torch.mean(batch_tmp.x[:, 0])))
                    gs_rela_perm, _ = uti_func.calc_rela_perm(args, comp_args, tmp_data.x[:, 0], 1. - tmp_data.x[:, 0])  # calculate gs rela perm
                    #print(gs_rela_perm.shape)
                    #print(torch.mean(gs_rela_perm))
                    tmp_data.x[:, -1] = gs_rela_perm              # update cell-wise rela perm

                viz_data.y[:, num]= stats.unnormalize( pred.squeeze(), mean_vec_y[0], std_vec_y[0] )
                
                # gs_data - viz_data = error_data
                eval_data.y[:, num] = (viz_data.y[:, num] - data.y[:, num])
                if (args.var_type == 'sat'):
                    eps = 0.01
                    # sat relative difference 
                    rmse_var.append( torch.sqrt(torch.mean(eval_data.y[:, num]**2) ))  

                elif (args.var_type == 'p'):
                    eps = 0.0
                    # for pressure, we use rrmse (relative RMSE)
                    rmse_var.append( torch.sqrt(torch.mean(eval_data.y[:, num]**2))/ torch.mean(data.y[:, num]) ) # scaled by the average value of gs truth 
 
                relative_err_var.append(torch.mean(torch.abs(eval_data.y[:, num]/(data.y[:, num] + eps))))
                
                if (save_vtk):
                    sg_gs.append(data.y[:, num].numpy())
                    sg_pred.append(viz_data.y[:, num].numpy())
                    sg_eval.append(eval_data.y[:, num].numpy())

                # pred gives the learnt saturation changes between two timsteps  
              
            if (sequntial):
                del tmp_data
                del h_0
                del c_0
            #traj_num += 1

    if (save_vtk):
        data_file.create_dataset('sg_gs', data = np.array(sg_gs))
        data_file.create_dataset('sg_pred', data = np.array(sg_pred))
        data_file.create_dataset('sg_eval', data = np.array(sg_eval))
        data_file.create_dataset('sg_rmse', data = rmse_var)
        data_file.close()
        print('Finish writing {}'.format(file_name))

    if (make_movie):
        if (sequntial or rolling_out):
            # lstm 
            # one mesh; multiple ys
            make_animation_lstm(gs_data_loader, viz_data_loader, eval_data_loader, rmse_var, file_dir,
                          gif_name, args, skip, True, plot_rmse, traj_num)
        else:
            # one-step
            # multiple mesh; one y
            make_animation_onestep(gs_data_loader, viz_data_loader, eval_data_loader, rmse_var, file_dir,
                      gif_name,args, skip, True, plot_rmse, traj_num)

    return eval_data_loader, rmse_var, relative_err_var


if __name__ == '__main__':
    pass
