# Valid Findings - Quick Verification Sample

**10 verified findings** (3 critical + 7 high) with pattern diversity for fast manual verification.

## 1. ml-001 (critical)

**File:** 
**Repo:** [louzounlab__pygon](https://github.com/louzounlab/pygon)
**Location:** function `calculate_features` (line 238)
**Paper:** Planted Dense Subgraphs in Dense Random Graphs Can Be Recovered using Graph-based Machine Learning
**Authors:** Itay Levinas, yoram louzoun

**Issue:** ml-001: Issue detected

**Explanation:** Data leakage: scaler/encoder is fit on full data including test set. Model performance will be inflated. Use sklearn.pipeline.Pipeline so fitting happens inside each fold.



**Code:**
```python
def calculate_features(graphs, params, graph_params):
    adjacency_matrices, feature_matrices = [], []
    for graph in graphs:
        fc = FeatureCalculator(graph_params, graph, "", params['features'], dump=False, gpu=True, device=0)
        adjacency_matrices.append(fc.adjacency_matrix)
        feature_matrices.append(fc.feature_matrix)

                                                                                                                   
                                                                                                                     
                                
    scaler = StandardScaler()
    all_matrix = np.vstack(feature_matrices)
    scaler.fit(all_matrix)
    for i in range(len(feature_matrices)):
        feature_matrices[i] = scaler.transform(feature_matrices[i].astype('float64'))
    return adjacency_matrices, feature_matrices
```

**Verification reasoning:** VALID: The scaler is fit on all feature matrices combined before the train/test split occurs in `split_into_folds`, meaning test data statistics leak into the scaler fitted on the full dataset. The comment in the code even acknowledges this ("Having all the graphs regardless whether they are training, eval of test") but incorrectly dismisses it. Fitting the scaler on all data inflates model performance by allowing test set distribution information to influence normalization.

---

## 2. ml-010 (critical)

**File:** 
**Repo:** [BrunoScholles98__Deep-Learning-for-Bone-Health-Classification-through-X-ray-Imaging](https://github.com/BrunoScholles98/Deep-Learning-for-Bone-Health-Classification-through-X-ray-Imaging)
**Location:** function `main` (line 199)
**Paper:** Osteoporosis screening: Leveraging EfficientNet with complete and cropped facial panoramic radiography imaging
**Authors:** Bruno Scholles Soares Dias et al.

**Issue:** ml-010: Issue detected

**Explanation:** Multi-test leakage: No held-out test set. Validation set used for both tuning and final evaluation. Create a separate test set that is never used during model development.


**Code:**
```python
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    train_ds = TripletVolumeDataset(DATA_DIR, "train", get_tf(True))
    val_ds   = TripletVolumeDataset(DATA_DIR, "test",  get_tf(False))

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

    model     = M3T(in_ch=3, out_ch=OUT_CHANNELS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = GradScaler()

    run_dir = os.path.join(OUTPUT_ROOT,
                           datetime.now().strftime("M3T_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    cfg = {k:v for k,v in globals().items()
           if k.isupper() and isinstance(v,(int,float,str,bool))}
    json.dump(cfg, open(os.path.join(run_dir,"config.json"),"w"))

    best_f1, patience = 0., 0
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc, tr_f1, _ = run_epoch(model, train_ld, criterion,
                                             optimizer, scaler, device, True)
        vl_loss, vl_acc, vl_f1, _ = run_epoch(model, val_ld,   criterion,
                                             optimizer, scaler, device, False)

        log.info(f"[{epoch:02d}/{EPOCHS}] "
                 f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} tr_f1={tr_f1:.3f} | "
                 f"val_loss={vl_loss:.4f} val_acc={vl_acc:.3f} val_f1={vl_f1:.3f}")

        if vl_f1 > best_f1:
            best_f1, patience = vl_f1, 0
            torch.save(model.state_dict(), os.path.join(run_dir,"best.pth"))
        else:
            patience += 1
            if patience >= EARLY_STOP:
                log.info("Early stopping."); break

    log.info(f"Melhor F1 (val): {best_f1:.3f}")
```

**Verification reasoning:** VALID: The code only defines two datasets — `train` and `test` (used as validation) — with no separate held-out test set. The `test` split is used for both early stopping decisions and best model selection (`best_f1`), meaning final reported performance (`best F1 (val): ...`) is on data that influenced training decisions. This is classic test set leakage via multi-test contamination.

---

## 3. pt-015 (high)

**File:** 
**Repo:** [czq142857__NDC](https://github.com/czq142857/NDC)
**Location:** function `testing_section` (line 385)
**Paper:** Neural Dual Contouring
**Authors:** Zhiqin Chen et al.

**Issue:** pt-015: Issue detected

**Explanation:** torch.load() without map_location can fail on CPU-only systems. Always specify map_location='cpu' or map_location=device for portable model loading.


**Code:**
```python
print('loading net...')
    if net_bool and (FLAGS.method == "undc" or FLAGS.input_type != "sdf"):
        network_bool.load_state_dict(torch.load(FLAGS.checkpoint_dir+"/weights_"+FLAGS.method+"_"+FLAGS.input_type+"_bool.pth"))
        print('network_bool weights loaded')
    if net_float:
        network_float.load_state_dict(torch.load(FLAGS.checkpoint_dir+"/weights_"+FLAGS.method+"_"+FLAGS.input_type+"_float.pth"))
```

**Verification reasoning:** VALID: `torch.load()` is called without `map_location` on lines 385 and 388, which will fail when loading GPU-saved checkpoints on CPU-only systems. The code does use a `device` variable elsewhere (lines 405-407), confirming device-awareness exists, but the load calls don't use it. Fix: `torch.load(..., map_location=device)`.

---

## 4. rep-002 (high)

**File:** 
**Repo:** [louzounlab__pygon](https://github.com/louzounlab/pygon)
**Location:** function `train_pygon` (line 300)
**Paper:** Planted Dense Subgraphs in Dense Random Graphs Can Be Recovered using Graph-based Machine Learning
**Authors:** Itay Levinas, yoram louzoun

**Issue:** rep-002: Issue detected

**Explanation:** CUDA non-determinism: some GPU ops are non-deterministic by default. Set torch.use_deterministic_algorithms(True) and torch.backends.cudnn.benchmark = False.



**Code:**
```python
def train_pygon(training_features, training_adjs, training_labels, eval_features, eval_adjs, eval_labels,
                params, class_weights, activations, unary, coeffs, graph_params):
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    pygon = PYGONModel(n_features=training_features[0].shape[1], hidden_layers=params["hidden_layers"],
                       dropout=params["dropout"],
                       activations=activations, p=graph_params["probability"], normalization=params["edge_normalization"])
    pygon.to(device)
    opt = params["optimizer"](pygon.parameters(), lr=params["lr"], weight_decay=params["regularization"])

    n_training_graphs = len(training_labels)
    graph_size = graph_params["vertices"]
    n_eval_graphs = len(eval_labels)

    counter = 0                      
    min_loss = None
    for epoch in range(params["epochs"]):
                                                                        
        training_graphs_order = np.arange(n_training_graphs)
        np.random.shuffle(training_graphs_order)
        for i, idx in enumerate(training_graphs_order):
            training_mat = torch.tensor(training_features[idx], device=device)
            training_adj, training_lbs = map(lambda x: torch.tensor(data=x[idx], dtype=torch.double, device=device),
                                             [training_adjs, training_labels])
            pygon.train()
            opt.zero_grad()
            output_train = pygon(training_mat, training_adj)
            output_matrix_flat = (torch.mm(output_train, output_train.transpose(0, 1)) + 1 / 2).flatten()
            training_criterion = build_weighted_loss(unary, class_weights, training_lbs)
            loss_train = coeffs[0] * training_criterion(output_train.view(output_train.shape[0]), training_lbs) + \
                coeffs[1] * pairwise_loss(output_matrix_flat, training_adj.flatten()) + \
                coeffs[2] * binomial_reg(output_train, graph_params)
            loss_train.backward()
            opt.step()

                                                                          
        graphs_order = np.arange(n_eval_graphs)
        np.random.shuffle(graphs_order)
        outputs = torch.zeros(graph_size * n_eval_graphs, dtype=torch.double)
        output_xs = torch.zeros(graph_size ** 2 * n_eval_graphs, dtype=torch.double)
        adj_flattened = torch.tensor(np.hstack([eval_adjs[idx].flatten() for idx in graphs_order]))
        for i, idx in enumerate(graphs_order):
            eval_mat = torch.tensor(eval_features[idx], device=device)
            eval_adj, eval_lbs = map(lambda x: torch.tensor(data=x[idx], dtype=torch.double, device=device),
                                     [eval_adjs, eval_labels])
            pygon.eval()
            output_eval = pygon(eval_mat, eval_adj)
            output_matrix_flat = (torch.mm(output_eval, output_eval.transpose(0, 1)) + 1 / 2).flatten()
            output_xs[i * graph_size ** 2:(i + 1) * graph_size ** 2] = output_matrix_flat.cpu()
            outputs[i * graph_size:(i + 1) * graph_size] = output_eval.view(output_eval.shape[0]).cpu()
        all_eval_labels = torch.tensor(np.hstack([eval_labels[idx] for idx in graphs_order]), dtype=torch.double)
        eval_criterion = build_weighted_loss(unary, class_weights, all_eval_labels)
        loss_eval = (coeffs[0] * eval_criterion(outputs, all_eval_labels) +
                     coeffs[1] * pairwise_loss(output_xs, adj_flattened) +
                     coeffs[2] * binomial_reg(outputs, graph_params)).item()

        if min_loss is None:
            current_min_loss = loss_eval
        else:
            current_min_loss = min(min_loss, loss_eval)

        if epoch >= 10 and params["early_stop"]:                                             
            if min_loss is None:
                min_loss = current_min_loss
                torch.save(pygon.state_dict(), "tmp_time.pt")                        
            elif loss_eval < min_loss:
                min_loss = current_min_loss
                torch.save(pygon.state_dict(), "tmp_time.pt")                        
                counter = 0
            else:
                counter += 1
                if counter >= 40:                         
                    break
                                                                         
    pygon.load_state_dict(torch.load("tmp_time.pt"))
    os.remove("tmp_time.pt")
    return pygon
```

**Verification reasoning:** VALID: The `train_pygon` function uses CUDA without setting `torch.use_deterministic_algorithms(True)` or `torch.backends.cudnn.benchmark = False`, meaning GPU operations like `torch.mm` may produce non-deterministic results across runs. This is a real reproducibility concern for scientific code — results could vary between runs even with the same random seed. The function also uses `np.random.shuffle` without seeding, compounding the non-determinism issue.

---

## 5. num-005 (high)

**File:** 
**Repo:** [openclimatefix__graph_weather](https://github.com/openclimatefix/graph_weather)
**Location:** method `Era5Dataset.__init__` (line 123)
**Paper:** WeatherMesh-3: Fast and accurate operational global weather forecasting
**Authors:** Haoxing Du et al.

**Issue:** num-005: Issue detected

**Explanation:** Division by zero when normalization factor (std, range, or norm) is zero for constant/zero-valued data, producing inf or NaN. Add an epsilon guard or explicit zero check before dividing.



**Code:**
```python
def __init__(self, xarr, transform=None):
        """
        Arguments:
            #TODO
        """
        ds = np.asarray(xarr.to_array())
        ds = torch.from_numpy(ds)
        ds -= ds.min(0, keepdim=True)[0]
        ds /= ds.max(0, keepdim=True)[0]
        ds = rearrange(ds, "C T H W -> T (H W) C")
        self.ds = ds
```

**Verification reasoning:** VALID: The code performs min-max normalization at line 123 with `ds /= ds.max(0, keepdim=True)[0]`, but after subtracting the min (line 122), any constant channel will have `max == 0`, causing division by zero and producing `NaN` or `inf`. ERA5 climate data can absolutely contain constant-valued channels (e.g., land/sea masks, or channels with no variation in a selected time slice). There is no epsilon guard or zero check present.

---

## 6. pt-009 (high)

**File:** 
**Repo:** [BrunoScholles98__Deep-Learning-for-Bone-Health-Classification-through-X-ray-Imaging](https://github.com/BrunoScholles98/Deep-Learning-for-Bone-Health-Classification-through-X-ray-Imaging)
**Location:** function `train_by_one_epoch` (line 107)
**Paper:** Osteoporosis screening: Leveraging EfficientNet with complete and cropped facial panoramic radiography imaging
**Authors:** Bruno Scholles Soares Dias et al.

**Issue:** pt-009: Issue detected

**Explanation:** Loss tensor accumulation: Accumulating loss tensors directly keeps entire computation graphs in memory, causing memory leaks. Extract the scalar value using loss.item() or loss.detach().cpu().item() before accumulating.


**Code:**
```python
def train_by_one_epoch(model, criterion, optimizer, train_dl, all_steps_counter_train, writer):
    accuracy_fnc = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(DEVICE)
    mean_loss_train = 0
    train_epoch_accuracy = 0

    training_bar = tqdm.tqdm(enumerate(train_dl), total=len(train_dl))
    training_bar.set_description("Training Progress (Epoch)")

    for step_train, inp in training_bar:
        inputs, labels = inp
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        y_hat, loss = train_one_step(model, optimizer, criterion, inputs, labels)
        mean_loss_train += loss
        training_iteration_accuracy = accuracy_fnc(y_hat, labels)
        train_epoch_accuracy += training_iteration_accuracy
            
        if step_train % LOG_INTERVAL == 0:
            writer.add_scalar('Train/Iteration_Loss', loss, all_steps_counter_train)
            writer.add_scalar('Train/Iteration_Accuracy', training_iteration_accuracy, all_steps_counter_train)
        
        all_steps_counter_train += 1

    mean_loss_train /= len(train_dl)
    train_epoch_accuracy /= len(train_dl)
    
    return all_steps_counter_train, mean_loss_train, train_epoch_accuracy
```

**Verification reasoning:** VALID: The code accumulates `loss` (a PyTorch tensor with attached computation graph) directly via `mean_loss_train += loss` at line 111. Since `train_one_step` returns the raw loss tensor after `.backward()`, the computation graph is retained across all loop iterations, causing GPU/CPU memory to grow with each step. Fix: use `loss.item()` when accumulating.

---

## 7. ml-010 (critical)

**File:** 
**Repo:** [songtaoliu0823__crebm](https://github.com/songtaoliu0823/crebm)
**Location:** function `main` (line 90)
**Paper:** Preference Optimization for Molecule Synthesis with Conditional Residual Energy-based Models
**Authors:** Songtao Liu et al.

**Issue:** ml-010: Issue detected

**Explanation:** Multi-test leakage: No held-out test set. Validation set used for both tuning and final evaluation. Create a separate test set that is never used during model development.


**Code:**
```python
train_data = TensorDataset(train_fps, train_costs)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)


val_data = TensorDataset(val_fps, val_costs)
val_sampler = RandomSampler(val_data)
```

**Verification reasoning:** VALID: The code splits data 90/10 into train/val with no separate test set, and the validation loss is used to track `best_val_loss` for model selection (line 98). This means the validation set serves dual purpose: hyperparameter/epoch selection and final evaluation, causing optimistic bias in reported performance. A held-out test set that is never used during training or model selection is missing.

---

## 8. pt-015 (high)

**File:** 
**Repo:** [kuangdai__sofa](https://github.com/kuangdai/sofa)
**Location:** module `<module>` (line 98)
**Paper:** Deep Learning Evidence for Global Optimality of Gerver's Sofa
**Authors:** Kuangdai Leng et al.

**Issue:** pt-015: Issue detected

**Explanation:** torch.load() without map_location can fail on CPU-only systems. Always specify map_location='cpu' or map_location=device for portable model loading.


**Code:**
```python
print("Last area:", gg["area"].item())

    if largest_area >= 0.:
        model.load_state_dict(torch.load(f"{out_dir}/best_model.pt"))
        u1, u2 = model.forward(alpha)
        gg = compute_area(alpha, beta1, beta2, u1, u2,
                          n_areas=args.n_areas, return_geometry=True)
```

**Verification reasoning:** VALID: `torch.load()` is called without `map_location` on line 98. The model was saved after training (which used `args.device`), so loading on a CPU-only machine will fail if it was trained on GPU. The fix is trivial: `torch.load(f"{out_dir}/best_model.pt", map_location=args.device)`.

---

## 9. rep-002 (high)

**File:** 
**Repo:** [amitaysicherman__reactembed](https://github.com/amitaysicherman/reactembed)
**Location:** function `main` (line 77)
**Paper:** ReactEmbed: A Cross-Domain Framework for Protein-Molecule Representation Learning via Biochemical Reaction Networks
**Authors:** Amitay Sicherman, Kira Radinsky

**Issue:** rep-002: Issue detected

**Explanation:** CUDA non-determinism: some GPU ops are non-deterministic by default. Set torch.use_deterministic_algorithms(True) and torch.backends.cudnn.benchmark = False.



**Code:**
```python
def main(data_name, batch_size, p_model, m_model, shared_dim, n_layers, hidden_dim, dropout,
         epochs, lr, flip_prob=0, samples_ratio=1, no_pp_mm=0, datasets=None, override=False):
    name = model_args_to_name(p_model, m_model, n_layers, hidden_dim, dropout, epochs, lr, batch_size, flip_prob,
                              shared_dim, samples_ratio, no_pp_mm)
    save_dir = f"data/{data_name}/model/{name}/"
    model_file = f"{save_dir}/model.pt"
    if not override and os.path.exists(model_file):
        print("Model already exists")
        return

    os.makedirs(save_dir, exist_ok=True)
    if datasets is not None:
        train_loader, valid_loader, test_loader = datasets
    else:
        train_loader = get_loader(data_name, "train", batch_size, p_model, m_model, flip_prob, samples_ratio, no_pp_mm)
        valid_loader = get_loader(data_name, "valid", batch_size, p_model, m_model, flip_prob, samples_ratio, no_pp_mm)
        test_loader = get_loader(data_name, "test", batch_size, p_model, m_model, flip_prob, samples_ratio, no_pp_mm)

    p_dim = model_to_dim[p_model]
    m_dim = model_to_dim[m_model]

    model = build_models(p_dim, m_dim, shared_dim, n_layers, hidden_dim, dropout, save_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad), "parameters")

    contrastive_loss = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x1, x2: 1 - F.cosine_similarity(x1, x2))

    best_valid_loss = float("inf")
    for epoch in range(epochs):
        visualize_training(model, train_loader.dataset, device, epoch, save_dir)

        train_loss = run_epoch(model, optimizer, train_loader, contrastive_loss, is_train=True)
        with torch.no_grad():
            valid_loss = run_epoch(model, optimizer, valid_loader, contrastive_loss, is_train=False)
            test_loss = run_epoch(model, optimizer, test_loader, contrastive_loss, is_train=False)
        print(f"Epoch {epoch} Train Loss: {train_loss}, Valid Loss: {valid_loss}, Test Loss: {test_loss}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{save_dir}/model.pt")
            print("Model saved")

        with open(f"{save_dir}/losses.txt", "a") as f:
            f.write(f"Epoch {epoch}: Train Loss: {train_loss}, Valid Loss: {valid_loss}, Test Loss: {test_loss}\n")

    with open(f"{save_dir}/losses.txt", "a") as f:
        f.write(f"Best Valid Loss: {best_valid_loss}")
```

**Verification reasoning:** VALID: The `main` function trains a neural network with GPU operations but never sets `torch.use_deterministic_algorithms(True)` or `torch.backends.cudnn.benchmark = False`. The code uses operations like `TripletMarginWithDistanceLoss` with cosine similarity that can be non-deterministic on GPU, and there's no seed setting anywhere visible. This is a biology domain model where reproducibility matters for scientific validity.

---

## 10. rep-002 (high)

**File:** 
**Repo:** [zhixunlee__fairgb](https://github.com/zhixunlee/fairgb)
**Location:** function `run` (line 40)
**Paper:** Rethinking Fair Graph Neural Networks from Re-balancing
**Authors:** ZHIXUN LI et al.

**Issue:** rep-002: Issue detected

**Explanation:** CUDA non-determinism: some GPU ops are non-deterministic by default. Set torch.use_deterministic_algorithms(True) and torch.backends.cudnn.benchmark = False.



**Code:**
```python
def run(data, args):
    acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(
        args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)
    
    neighbor_dist_list = get_ins_neighbor_dist(data.y.size(0), data.edge_index, args.device)

    data = data.to(args.device)
    n_cls = data.y.max().int().item() + 1
    n_sen = data.sens.max().int().item() + 1
    index_list = torch.arange(len(data.y)).to(args.device)
    group_num_list, idx_info = [], []
    for i in range(n_cls):
        for j in range(n_sen):
            mask = ((data.y == i) & (data.sens == j) & data.train_mask)
            data_num = mask.sum()
            group_num_list.append(int(data_num.item()))
            idx_info.append(index_list[mask])

    encoder, classifier, optimizer_e, optimizer_c = get_enc_cls_opt(args)

    for count in range(args.runs):
        seed_everything(count + args.seed)
        classifier.reset_parameters()
        encoder.reset_parameters()

        best_val_tradeoff = 0
        for epoch in range(0, args.epochs):
            encoder.train()
            classifier.train()

            optimizer_c.zero_grad()
            optimizer_e.zero_grad()
            
            sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(group_num_list, idx_info, args.eta)
            beta = torch.distributions.beta.Beta(2, 2)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            lam = lam.to(args.device)
        
            if epoch >= args.warmup:
                new_edge_index = neighbor_sampling(data.x.size(0), data.edge_index, sampling_src_idx, neighbor_dist_list)
                new_x = saliency_mixup(data.x, sampling_src_idx, sampling_dst_idx, lam)
                
                h = encoder(new_x, new_edge_index)
                output = classifier(h)

                add_num = output.shape[0] - data.train_mask.shape[0]
                new_train_mask = torch.ones(add_num, dtype=torch.bool, device=args.device)
                new_train_mask = torch.cat((torch.zeros(data.train_mask.shape[0], dtype=torch.bool, device=args.device), new_train_mask), dim=0)

                loss_src = F.binary_cross_entropy_with_logits(
                    output[new_train_mask], data.y[sampling_src_idx].unsqueeze(1).to(args.device), reduction='none')
                loss_dst = F.binary_cross_entropy_with_logits(
                    output[new_train_mask], data.y[sampling_dst_idx].unsqueeze(1).to(args.device), reduction='none')
                
                pos_grad_src = (1. - torch.exp(-loss_src).detach()) * lam
                pos_grad_dst = (1. - torch.exp(-loss_dst).detach()) * (1-lam)
                grad_count = []
                for i in range(n_cls):
                    for j in range(n_sen):
                        mask_src = (data.y[sampling_src_idx] == i) & (data.sens[sampling_src_idx] == j)
                        mask_dst = (data.y[sampling_dst_idx] == i) & (data.sens[sampling_dst_idx] == j)
                        grad_count.append(pos_grad_src[mask_src].sum().item() + pos_grad_dst[mask_dst].sum().item())

                min_grad = np.min(grad_count)
                group_weight_list = [float(min_grad)/(float(num) + EPS) for num in grad_count]

                for i in range(n_cls):
                    for j in range(n_sen):
                        mask_src = (data.y[sampling_src_idx] == i) & (data.sens[sampling_dst_idx] == j)
                        mask_dst = (data.y[sampling_dst_idx] == i) & (data.sens[sampling_dst_idx] == j)
                        loss_src[mask_src] *= group_weight_list[i*2+j]
                        loss_dst[mask_dst] *= group_weight_list[i*2+j]

                loss = lam * loss_src + (1-lam) * loss_dst
                loss.mean().backward()
            else:
                h = encoder(data.x, data.edge_index)
                output = classifier(h)

                loss_c = F.binary_cross_entropy_with_logits(
                    output[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(args.device))
                loss_c.backward()

            optimizer_e.step()
            optimizer_c.step()

            accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged3(classifier, encoder, data)

            if epoch % 10 == 0:
                print("RUN: {}/{}, Epoch: {:04}/{:04} | Val Acc: {:.4f}, Test Acc: {:.4f}, Test AUC: {:.4f}, Test F1: {:.4f}, Test SP: {:.4f}, Test EO: {:.4f}".format(
                    count+1, args.runs, epoch, args.epochs, accs['val'], accs['test'], auc_rocs['test'], F1s['test'], tmp_parity['test'], tmp_equality['test']
                ))

            if (auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff):
                test_acc = accs['test']
                test_auc_roc = auc_rocs['test']
                test_f1 = F1s['test']
                test_parity, test_equality = tmp_parity['test'], tmp_equality['test']

                best_val_tradeoff = auc_rocs['val'] + F1s['val'] + \
                    accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val'])
                                
                print("\033[0;30;41m RUN: {}/{}, Epoch: {:04}/{:04} | Val Acc: {:.4f}, Test Acc: {:.4f}, Test AUC: {:.4f}, Test F1: {:.4f}, Test SP: {:.4f}, Test EO: {:.4f}\033[0m".format(
                    count+1, args.runs, epoch, args.epochs, accs['val'], accs['test'], auc_rocs['test'], F1s['test'], tmp_parity['test'], tmp_equality['test']
                ))

        acc[count] = test_acc
        f1[count] = test_f1
        auc_roc[count] = test_auc_roc
        parity[count] = test_parity
        equality[count] = test_equality

    return acc, f1, auc_roc, parity, equality
```

**Verification reasoning:** VALID: The code uses `seed_everything(count + args.seed)` for seeding but never sets `torch.use_deterministic_algorithms(True)` or `torch.backends.cudnn.benchmark = False`. Graph neural network operations (especially neighborhood sampling and aggregation) are known to have non-deterministic CUDA implementations. This is a fairness research codebase where reproducibility is critical — non-deterministic GPU ops could produce different fairness metrics across runs even with the same seed.

---
