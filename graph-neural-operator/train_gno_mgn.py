
def train(train_loader, 
          test_loader_list,
          u_normalizer, 
          device, 
          args, 
          kernel_nn_class, 
          PATH=None):

    # Initialize a new wandb run
    wandb.init(project="gno_project", entity="ju1", name=args.model + '_kernel_width' + \
                                               str(args.ker_width) + '_width_'+ str(args.width)+ \
                                                '_depth' + str(args.depth) )
    # Optionally, define the configuration parameters
    wandb.config.update(args)
    
    if args.use_kernel:
        model = kernel_nn_class(args.width,
                     args.ker_width,
                     args.depth,
                     args.edge_features,
                     args.node_features).to(device)
    else:
        # MGN-like model
        model = kernel_nn_class(args.node_features,
                     args.edge_features,
                     args.width,
                     1, args).to(device)

    # Optionally, log model summary
    gnn_model_summary(model, args, args.model + '_depth' + str(args.depth))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    
    myloss = LpLoss(size_average=False)
    u_normalizer.cuda()

    best_test_loss = np.inf
    model.train()

    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        train_mse = 0.0
        train_l2 = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            mse = F.mse_loss(out.view(-1, 1), batch.y.view(-1,1))
            # mse.backward()
            loss = torch.norm(out.view(-1) - batch.y.view(-1),1)
            loss.backward()

            l2 = myloss(u_normalizer.decode(out.view(args.batch_size,-1)), u_normalizer.decode(batch.y.view(args.batch_size, -1)))
            # l2.backward()
            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()
        train_loss = train_l2 / (args.ntrain * args.k)

        # Log training metrics to wandb
        #wandb.log({"epoch": epoch, "train_loss": train_loss, "train_mse": train_mse})

        # Test the model periodically and log the metrics
        if epoch % 10 == 0:
            test_loss_array = test(test_loader_list, args, device, model, u_normalizer, myloss)
            test_loss = np.sum(test_loss_array)
            # Log the max test loss to wandb
            #wandb.log({"epoch": epoch, "combined_test_loss": test_loss})

            # Log each individual test loss
            wandb.log({
                "epoch": epoch,
                "test_loss_1": test_loss_array[0],
                "test_loss_2": test_loss_array[1],
                "test_loss_3": test_loss_array[2]
            })

            # Save the model checkpoint with wandb
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = copy.deepcopy(model)
                wandb.run.summary["best_test_loss"] = best_test_loss

                # Save the model using the wandb save function
                if PATH is not None:
                    wandb.save(PATH)
        
        # Log the training and test loss to wandb
        wandb.log({"train_loss": round(train_loss, 2), "test_loss": round(test_loss.item(), 2)})
        
        if(epoch%100==0):
            #print("train loss", str(round(total_loss,2)), "test loss", str(round(test_loss.item(),2)))
            if(args.save_best_model):
                PATH = os.path.join(args.checkpoint_dir, args.model + '.pt')
                torch.save(best_model.state_dict(), PATH )
                

    wandb.finish()  # Finish the wandb run
    
    return best_model, best_test_loss


def test(test_loader_list, args, device, model, u_normalizer, myloss):
    '''
    Performs a test loop on the dataset for GNO.
    Explain input arguments:
    test_loader: test dataset
    device: GPU or CPU
    test_model: trained model
    myloss: loss function
    u_normalizer: normalizer for output
    '''
    
    u_normalizer.cuda()
    model = model.to(device)
    ntest = args.ntest
    test_res_size = len(test_loader_list)
    test_loss_array = np.zeros([test_res_size])
    batch_size2 = args.batch_size2
    with torch.no_grad():
        for i in range(3):
            test_loss = 0.0
            for batch in test_loader_list[i]:
                batch.to(device)
                out = model(batch)
                test_l2 = myloss(u_normalizer.decode(out.view(batch_size2,-1)),
                                     batch.y.view(batch_size2, -1))
                test_loss += test_l2.item()
            test_loss_array[i] = test_loss / ntest

    return test_loss_array

