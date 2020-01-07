####################################################################################
## Supervised machine learning
####################################################################################

# we permute the farm to be tested, so that we train a model on all the others farms and test performance. 

sml_compo_hyper_fit <- function(data, comp, index, algo, grid, optim_overfit = F) {
  
  ## library
  require(ranger)
  require(caret)
  require(kohonen)
  require(irr)
  #require(mxnet)
  require(vegan)
  require(e1071)
  require(foreach)
  
  fitControl <- trainControl(   ## 10-fold CV
    method = "repeatedcv",
    number = 10,
    ## repeated ten times
    repeats = 2,
    search="random")
  
  
  OTU <- data.frame(data) #data.frame(otu_aqua_train)
  COMP <- comp # comp_aqua_train
  
  ### the index to fit on
  index <- index
  
  grid_se <- grid
  
  #### callback function for mxnet
  mx.callback.early.stop <- function(log, farm, grid) {
    function(iteration, nbatch, env, verbose) {
      if (!is.null(env$metric)) {
        RMSE <- env$metric$get(env$train.metric)$value
        log$train <- c(log$train , RMSE)
        if (!is.null(log$train)) {
          if (length(log$train) > 5) {
            diff <- abs(mean(tail(log$train, 5)))
            diffSD <- abs(sd(tail(log$train, 5)))
            print(paste(i, " ", j, "hyp.par./", dim(grid_se)[1], " - epoch :", iteration, "-> rmse :", diff, "-", log$train[iteration], "-> sd :", diffSD))
            if (is.na(diff) != TRUE) {
              if (diff > log$train[iteration] -0.05 & diff < log$train[iteration] +0.05 & diffSD < 0.05) {
                print(paste(i, " ", j, "hyp.par./", dim(grid_se)[1], " - STOP  :", iteration, "-> rmse :", diff, "-", log$train[iteration], "-> sd :", diffSD))
                return(FALSE)
              }
            } else if (is.na(diff) == TRUE) {
              return(FALSE)
            }
          }
        }
      }
      return(TRUE)
    }
  }
  
  
  ### algorithm for the fitting
  # random forest - ranger package "RF"
  # SOM - caret package "SOM" TO BE DONE
  # deep learning - mxnet "DL"
  algo <- algo
  ### check whether or not we overfit T or F
  # number of trees for random forest
  # iteration for mxnet
  # 
  
  ## test of right length between data and compo
  if (dim(OTU)[[1]] != length(COMP[,index])) print("No same size between datasets, check it out...")
  
  classification <- F
  
  farms <- unique(COMP$Locality) # [-length(unique(COMP$Locality))]
  #farms <- c("Aukrasanden","Beitveitnes","Bjorlykkestranda","Bjornsvik","Bremnessvaet","Brettingen","Kornstad","Nedre Kvarv","Rundreimstranda","Storvika")
  #farms <- c("Bjornsvik","Nedre Kvarv","Beitveitnes","Storvika", "Aukrasanden")
  
  RMSE <- c(1:dim(grid_se)[1])
  farm_rmse <- array(NA, c(length(RMSE), length(farms)))
  colnames(farm_rmse) <- farms
  RMSE <- cbind(grid_se, farm_rmse)
  
  combined1_rf <- c()
  combined2_rf <- c()
  combined1_sm <- c()
  combined2_sm <- c()
  combined1_nn <- c()
  combined2_nn <- c()
  combined1_lm <- c()
  combined2_lm <- c()
  
  combined1_dn <- c()
  combined2_dn <- c()
  
  combined1_sv <- c()
  combined2_sv <- c()
  
  ## gather RMSE for optim and check overfit
  over <- array(NA, c(2,10))
  rownames(over) <- c("training RMSE", "testing RMSE")
  colnames(over) <- c(10,50,100,150,200,250,300,350,400,500)
  
  farm_nam <- c()
  increm <- 0
  
  for (k in farms)
  {
    # i will be the farm to be assayed, so search for parameter for the remaining farms_ with CV
    farms_ <- subset(farms, farms != k)
    ## subsetting for assay
    nam_t_a <- paste("ass_farm_a_m", k,sep="_")
    nam_comp_t_a <- paste("ass_farm_c_a_m", k,sep="_")
    assign(nam_t_a, subset(OTU, COMP$Locality ==k))
    assign(nam_comp_t_a, subset(COMP, COMP$Locality ==k))
    ## subsetting for final training
    nam_a <- paste("ass_farm_m", k,sep="_")
    nam_comp_a <- paste("ass_farm_c_m", k,sep="_")
    assign(nam_a, subset(OTU, COMP$Locality !=k))
    assign(nam_comp_a, subset(COMP, COMP$Locality !=k))
    
    for (i in farms_)
    {
      
      # subsetting
      nam_t <- paste("test_farm_m", i,sep="_")
      nam_comp_t <- paste("test_farm_c_m", i,sep="_")
      assign(nam_t, subset(get(nam_a), get(nam_comp_a)$Locality ==i))
      assign(nam_comp_t, subset(get(nam_comp_a), get(nam_comp_a)$Locality ==i))
      
      nam <- paste("train_farm_m", i,sep="_")
      nam_comp <- paste("train_farm_c_m", i,sep="_")
      assign(nam, subset(get(nam_a), get(nam_comp_a)$Locality !=i))
      assign(nam_comp, subset(get(nam_comp_a), get(nam_comp_a)$Locality !=i))
      
      ## if random forest 
      if (algo == "RF")
      {
        ### if optim and check for overfit 
        if (optim_overfit == T) 
        {
          ## gather RMSE for optim and check overfit
          over <- array(NA, c(2,10))
          rownames(over) <- c("training RMSE", "testing RMSE")
          colnames(over) <- c(10,50,100,150,200,250,300,350,400,500)
          for (opt in colnames(over))
          {
            mod <- paste("RF_farm_m", i,sep="_")
            set.seed(1)
            assign(mod,ranger(get(nam_comp)[,index] ~ ., data=get(nam), mtry=floor(dim(get(nam))[2]/3), classification = classification, num.trees = as.numeric(opt), importance= "impurity", write.forest = T))
            ## prediction for new data
            predict_tr_rf <- predict(get(mod), get(nam_t))
            ## paste in over the RMSE values
            over["training RMSE",opt] <- get(mod)$prediction.error
            over["testing RMSE",opt] <- sqrt(mean((get(nam_comp_t)[,index] - predict_tr_rf$predictions)^2))
          }
          ### then get the minimum testing RMSE to get the optim num.tree param
          best_over <- as.numeric(attributes(which.min(over["testing RMSE",]))$names)
          print("num.tree effect on testing RMSE")
          print(over)
          print(paste("Using: ", best_over))
          # maybe export a plot and curves for overfitting check...
          mod <- paste("RF_farm_m", i,sep="_")
          set.seed(1)
          assign(mod,ranger(get(nam_comp)[,index] ~ ., data=get(nam), mtry=floor(dim(get(nam))[2]/3), classification = classification, num.trees = best_over, importance= "impurity", write.forest = T))
          ## prediction for new data
          predict_tr_rf <- predict(get(mod), get(nam_t))
          combined1_rf <- c(combined1_rf,predict_tr_rf$predictions)
          combined2_rf <- c(combined2_rf,get(nam_comp_t)[,index])
        } else 
        {
          # ## now the fitting, random forest with Ranger package with default mtry (1/3) for regression according to Breiman

          #assign(mod,ranger(get(nam_comp)[,index] ~ ., data=get(nam), mtry=floor(dim(get(nam))[2]/3), classification = classification, num.trees = 300, importance= "impurity", write.forest = T))
          ## first fit the models 
          print("Fitting RF and predictions...")
          for (j in 1:dim(grid_se)[1])
          {
            mod <- paste("RF_farm_m", j,sep="_")
            set.seed(1)
            assign(mod,ranger(get(nam_comp)[,index] ~ ., data=get(nam), mtry= floor(dim(get(nam))[2]*grid_se[j,"mtry"]), 
                            num.trees = 300, splitrule = grid_se[j,"splitrule"], min.node.size = grid_se[j,"min.node.size"], 
                            importance= "impurity", write.forest = T))
            ## prediction for new data
            preds <- predict(get(mod), as.matrix(get(nam_t)))
            ## paste the RMSE values
            RMSE[j,i] <- sqrt(mean((get(nam_comp_t)[,index] - preds$predictions)^2))
            print(paste(j, " is done -------- for ", i, RMSE[j,i]))
          }
          print("RF done...")
        }
      }
      
      
      ### if Self-Oragnizing Map with kohonen package
      if (algo =="SOM")
      {
        ## removing empty OTUs
        #OTUtr <- get(nam)[,colSums(get(nam)) != 0]
        # remove those ones for the testing dataset also
        #OTUte <- get(nam_t)[,colSums(get(nam)) != 0]

        ## parallel with foreach
        ## first fit the models 
        print("Fitting all the SOM models first...")
        set.seed(1)
        mod_list <- foreach(j = 1:dim(grid_se)[1]) %dopar% assign(paste("SOM_farm_m", i,j, sep="_"), bdk(as.matrix(get(nam)), get(nam_comp)[,index], rlen = 100, grid=somgrid(xdim = grid_se[j,"xdim"], ydim = grid_se[j,"ydim"], topo=grid_se[j,"topo"]), alpha = c(0.05, 0.01), radius = grid_se[j,"radius"], xweight = grid_se[j,"xweight"]))
        print("Fitting done.")
        ## prediction for new data
        for (j in 1:dim(grid_se)[1])
        {
          ## fetch the model 
          mod <- mod_list[[j]]
          ## prediction for new data
          preds <- predict(mod, as.matrix(get(nam_t)))
          ## paste the RMSE values
          RMSE[j,i] <- sqrt(mean((get(nam_comp_t)[,index] - preds$prediction)^2))
          print(paste(j, " is done -------- for ", i, RMSE[j,i]))
          
        }
        
      }
      
      ### if Support Vector Machine with e1071 package
      if (algo =="SVM")
      {
        ## removing empty OTUs
        #OTUtr <- get(nam)[,colSums(get(nam)) != 0]
        # remove those ones for the testing dataset also
        #OTUte <- get(nam_t)[,colSums(get(nam)) != 0]
        ## parallel with foreach
        ## first fit the models 
        print("Fitting all the SVM models first...")
        set.seed(1)
        mod_list <- foreach(j = 1:dim(grid_se)[1]) %dopar% assign(paste("SVM_farm_m", i,j, sep="_"), svm(as.matrix(get(nam)),
                         get(nam_comp)[,index], type = grid_se[j,"type"], kernel = grid_se[j,"kernel"],
                         epsilon=grid_se[j,"epsilon"], tolerance = grid_se[j,"tolerance"]))
        print("Fitting done.")
        ## prediction for new data
        for (j in 1:dim(grid_se)[1])
        {
          ## fetch the model 
          mod <- mod_list[[j]]
          ## prediction for new data
          preds <- predict(mod, get(nam_t))
          ## paste the RMSE values
          RMSE[j,i] <- sqrt(mean((get(nam_comp_t)[,index] - preds)^2))
          print(paste(j, " is done -------- for ", i, RMSE[j,i]))
          
        }
        
      }
      
      
      ### if DL with mxnet
      if (algo =="DL")
      {
        ## RMSE
        for (j in 1:dim(grid_se)[1])
        {
          mod <- paste("DeepNet_farm_m", i,sep="_")
          
          log <- mx.metric.logger$new()
          mx.set.seed(1)
          assign(mod,mx.mlp(as.matrix(get(nam)), get(nam_comp)[,index], verbose = F, epoch.end.callback = mx.callback.early.stop(log), dropout= grid_se[j,"dropout"], momentum=grid_se[j,"momentum"], array.layout="rowmajor", learning.rate=grid_se[j,"learning.rate"],hidden_node=grid_se[j,"hidden_node"], out_node=grid_se[j,"out_node"], num.round=100, activation=as.character(grid_se[j,"activation"]), out_activation='rmse', eval.metric=mx.metric.rmse))
          ## prediction for new data
          preds <- predict(get(mod), as.matrix(get(nam_t)), array.layout="rowmajor")
          ## paste in over the RMSE values
          RMSE[j,i] <- sqrt(mean((get(nam_comp_t)[,index] - preds)^2))
          print(paste(j, " is done -------- for ", i, RMSE[j,i]))
          
        }
        
      }
    }
    
    
    ## now do the prediction of the hold-out farm with best hyperparameter
    if (algo =="RF")
    {
      hyp <- RMSE[which.min(rowMeans(RMSE[,as.character(farms_)])),1:dim(grid_se)[2]]
      print("fitting RF with hyper-param :")
      print(hyp)
      mod <- ranger(get(nam_comp_a)[,index] ~., get(nam_a), mtry= floor(dim(get(nam))[2]*hyp[,"mtry"]), 
                    num.trees = 300, splitrule = hyp[,"splitrule"], min.node.size = hyp[,"min.node.size"], 
                    importance= "impurity", write.forest = T)
      
      ## prediction for new data
      preds <- predict(mod, as.matrix(get(nam_t_a)))
      ### concatenate the values over the k farms
      combined1_rf <- c(combined1_rf, preds$prediction)
      print(paste(k, " is assayed"))
      print(combined1_rf)
    }
    if (algo =="SOM")
    {
      hyp <- RMSE[which.min(rowMeans(RMSE[,as.character(farms_)])),1:dim(grid_se)[2]]
      print("fitting SOM with hyper-param :")
      print(hyp)
      mod <- bdk(as.matrix(get(nam_a)), get(nam_comp_a)[,index], rlen = 100, grid=somgrid(xdim = hyp[,"xdim"], ydim = hyp[,"ydim"], topo=hyp[,"topo"]), alpha = c(0.05, 0.01), radius = hyp[,"radius"], xweight = hyp[,"xweight"])
      ## prediction for new data
      preds <- predict(mod, as.matrix(get(nam_t_a)))
      ### concatenate the values over the k farms
      combined1_sm <- c(combined1_sm, preds$prediction)
      print(paste(k, " is assayed"))
      print(combined1_sm)
    }
    if (algo =="SVM")
    {
      hyp <- RMSE[which.min(rowMeans(RMSE[,as.character(farms_)])),1:dim(grid_se)[2]]
      print("fitting SVM with hyper-param :")
      print(hyp)
      set.seed(1)
      mod <- svm(as.matrix(get(nam_a)), get(nam_comp_a)[,index], type = hyp[,"type"], kernel = hyp[,"kernel"],
                 epsilon=hyp[,"epsilon"], tolerance = hyp[,"tolerance"])
      
      ## prediction for new data
      preds <- predict(mod, get(nam_t_a))
      ### concatenate the values over the k farms
      combined1_sv <- c(combined1_sv, preds)
      print(paste(k, " is assayed"))
      print(combined1_sv)
    }
    if (algo =="DL")
    {
      hyp <- RMSE[which.min(rowMeans(RMSE[,as.character(farms_)])),1:dim(grid_se)[2]]
      print("fitting Mxnet with hyper-param :")
      print(hyp)
      log <- mx.metric.logger$new()
      mx.set.seed(1)
      mod <- mx.mlp(as.matrix(get(nam_a)), get(nam_comp_a)[,index], verbose = F, epoch.end.callback = mx.callback.early.stop(log), dropout= hyp[,"dropout"], momentum=hyp[,"momentum"], array.layout="rowmajor", learning.rate=hyp[,"learning.rate"],hidden_node=hyp[,"hidden_node"], out_node=hyp[,"out_node"], num.round=100, activation=as.character(hyp[,"activation"]), out_activation='rmse', eval.metric=mx.metric.rmse)
      ## prediction for new data
      preds <- predict(mod, as.matrix(get(nam_t_a)), array.layout="rowmajor")
      ### concatenate the values over the k farms
      combined1_dn <- c(combined1_dn, preds)
      print(paste(k, " is assayed"))
      print(combined1_dn)
    }
  }
  
  if (algo =="RF")  { return(list("preds" = combined1_rf, "RMSE"= RMSE)) }
  if (algo =="SVM") { return(list("preds" = combined1_sv, "RMSE"= RMSE)) }
  if (algo =="SOM") { return(list("preds" = combined1_sm, "RMSE"= RMSE)) }
  if (algo =="DL")  { return(list("preds" = combined1_dn, "RMSE"= RMSE)) }
}


