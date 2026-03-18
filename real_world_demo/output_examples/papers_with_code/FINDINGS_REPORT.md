# Real-World Scientific ML Code Analysis Report (Verified Findings Only)

Analysis of Python code from **AI applications to scientific domains**. Papers sourced from PapersWithCode, filtered to include only scientific domains (biology, chemistry, physics, medicine, earth science, astronomy, materials, etc.) where ML/AI is applied to scientific discovery and domain-specific research.

## Analysis Summary

- **Analysis Date:** 2026-03-17 23:14
- **Report Generated:** 2026-03-17 20:58
- **scicode-lint Version:** 0.2.2
- **Papers with Findings:** 27 / 32 (84.4%)
- **Repos with Findings:** 29 / 35 (82.9%)
- **Files Analyzed:** 120 / 120
- **Files with Findings:** 69 (57.5%)
- **Total Findings:** 137

## Prefilter Summary

Files were classified by LLM as self-contained ML pipelines vs code fragments. Only self-contained files (complete training/inference workflows) were analyzed.

| Classification | Files | % |
|----------------|-------|---|
| Self-contained (analyzed) | 120 | 13.6% |
| Fragment (skipped) | 749 | 84.7% |
| Error | 15 | 1.7% |
| **Total** | **884** | |

### Papers/Repos Filtered

- **Papers:** 32 analyzed / 38 original (6 dropped)
- **Repos:** 35 analyzed / 47 original (12 dropped)

**12 repos dropped** (all files classified as fragments):

- `AstraZeneca__chemicalx`
- `CompRhys__aviary`
- `CompRhys__roost`
- `citrineinformatics-erd-public__piml_glass_forming_ability`
- `coarse-graining__cgnet`
- `idea-iitd__greed`
- `idea-iitd__neurosed`
- `mtian8__gw_spatiotemporal_gnn`
- `sheoyon-jhin__contime`
- `tpospisi__DeepCDE`
- ... and 2 more

*Prefilter model: qwen3-8b-fp8*

## Papers by Severity

Papers with at least one finding of each severity level (a paper may appear in multiple rows):

| Severity | Papers | % of Papers Analyzed |
|----------|--------|----------------------|
| Critical | 12 | 37.5% |
| High | 23 | 71.9% |
| Medium | 17 | 53.1% |

*Total papers analyzed: 32*

## Findings Distribution (per paper)

| Metric | All | Critical | High | Medium | Low |
|--------|-----|----------|------|--------|-----|
| Papers | 27 | 12 | 23 | 17 | 0 |
| Min | 1 | 1 | 1 | 1 | - |
| Max | 19 | 6 | 12 | 9 | - |
| Mean | 5.1 | 1.8 | 3.3 | 2.3 | - |
| Median | 3.0 | 1.0 | 2.0 | 2.0 | - |
| Std Dev | 4.9 | 1.5 | 3.1 | 2.1 | - |

## Verification Summary

**Overall Precision:** 62.0% (85 valid / 137 verified)

| Status | Count | % |
|--------|-------|---|
| Valid (confirmed) | 85 | 62.0% |
| Invalid (false positive) | 45 | 32.8% |
| Uncertain | 7 | 5.1% |

### Verified Findings by Severity

| Severity | Total | Valid | Invalid | Uncertain | Pending | Precision |
|----------|-------|-------|---------|-----------|---------|-----------|
| Critical | 21 | 5 | 16 | 0 | 0 | 24% |
| High | 77 | 52 | 18 | 7 | 0 | 68% |
| Medium | 39 | 28 | 11 | 0 | 0 | 72% |

## Findings by Scientific Domain

| Domain | Files Analyzed | With Findings | Finding Rate | Total Findings |
|--------|---------------|---------------|--------------|----------------|
| chemistry | 45 | 22 | 48.9% | 33 |
| earth_science | 17 | 10 | 58.8% | 19 |
| materials | 5 | 5 | 100.0% | 19 |
| none | 8 | 5 | 62.5% | 15 |
| medicine | 8 | 6 | 75.0% | 14 |
| biology | 7 | 5 | 71.4% | 11 |
| engineering | 10 | 6 | 60.0% | 10 |
| economics | 6 | 4 | 66.7% | 9 |
| astronomy | 5 | 2 | 40.0% | 3 |
| mathematics | 2 | 2 | 100.0% | 2 |
| physics | 2 | 1 | 50.0% | 1 |
| social_science | 1 | 1 | 100.0% | 1 |
| neuroscience | 3 | 0 | 0.0% | 0 |

## Findings by Category

| Category | Count | Unique Files | Unique Repos |
|----------|-------|--------------|--------------|
| scientific-reproducibility | 54 | 39 | 22 |
| ai-inference | 37 | 31 | 12 |
| ai-training | 25 | 18 | 13 |
| scientific-performance | 13 | 11 | 10 |
| scientific-numerical | 8 | 5 | 3 |

## Findings by Severity

| Severity | Count | % of Total |
|----------|-------|------------|
| Critical | 21 | 15.3% |
| High | 77 | 56.2% |
| Medium | 39 | 28.5% |

## Most Common Patterns

| Pattern | Category | Count | Files | Repos | Avg Confidence |
|---------|----------|-------|-------|-------|----------------|
| rep-002 | scientific-reproducibility | 21 | 21 | 13 | 95% |
| pt-015 | ai-inference | 19 | 19 | 8 | 95% |
| rep-004 | scientific-reproducibility | 11 | 11 | 6 | 95% |
| rep-006 | scientific-reproducibility | 10 | 10 | 7 | 85% |
| pt-013 | ai-inference | 9 | 9 | 5 | 95% |
| rep-003 | scientific-reproducibility | 7 | 7 | 6 | 95% |
| pt-007 | ai-inference | 5 | 5 | 3 | 95% |
| ml-010 | ai-training | 5 | 5 | 4 | 95% |
| num-005 | scientific-numerical | 4 | 4 | 2 | 95% |
| pt-004 | ai-training | 3 | 3 | 2 | 95% |

## Example Findings

Representative findings from each category (with links to source):

### ai-inference

**pt-015** (high, 95% confidence)

- **Repo:** songtaoliu0823__crebm (chemistry)
- **Location:** module `<module>` (line 67)
- **Paper:** Preference Optimization for Molecule Synthesis with Conditional Residual Energy-based Models
- **Authors:** Songtao Liu et al.
- **Issue:** pt-015: Issue detected
- **Explanation:** torch.load() without map_location can fail on CPU-only systems. Always specify map_location='cpu' or map_location=device for portable model loading.

- **Suggestion:** Review the code and fix according to the explanation.

```python
np.random.seed(args.seed)
random.seed(args.seed)

              
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ValueMLP(
        n_layers=args.n_layers,
```

**pt-007** (high, 95% confidence)

- **Repo:** rose-stl-lab__spherical-dyffusion (earth_science)
- **Location:** method `validate_one_epoch` (line 311)
- **Paper:** Probabilistic Emulation of a Global Climate Model with Spherical DYffusion
- **Authors:** Salva Rühling Cachay et al.
- **Issue:** pt-007: Issue detected
- **Explanation:** Missing model.eval() leaves dropout active and batchnorm using batch statistics instead of learned running statistics, producing incorrect inference results.

- **Suggestion:** Review the code and fix according to the explanation.

```python
def validate_one_epoch(self):
        aggregator = OneStepAggregator(
            self.train_data.area_weights.to(fme.get_device()),
            self.train_data.sigma_coordinates,
            self.train_data.metadata,
        )

        with torch.no_grad(), self._validation_context():
            for batch in self.valid_data.loader:
                stepped = self.stepper.run_on_batch(
                    batch.data,
                    optimization=NullOptimization(),
                    n_forward_steps=self.config.n_forward_steps,
                    aggregator=NullAggregator(),
                )
                stepped = compute_stepped_derived_quantities(stepped, self.valid_data.sigma_coordinates)
                aggregator.record_batch(
                    loss=stepped.metrics["loss"],
                    target_data=stepped.target_data,
                    gen_data=stepped.gen_data,
                    target_data_norm=stepped.target_data_norm,
                    gen_data_norm=stepped.gen_data_norm,
                )
        return aggregator.get_logs(label="val")
```

**pt-019** (medium, 95% confidence)

- **Repo:** rose-stl-lab__spherical-dyffusion (earth_science)
- **Location:** function `main` (line 388)
- **Paper:** Probabilistic Emulation of a Global Climate Model with Spherical DYffusion
- **Authors:** Salva Rühling Cachay et al.
- **Issue:** pt-019: Issue detected
- **Explanation:** cudnn.benchmark=True with variable input sizes causes overhead. Disable it for inference with varying batch sizes or dimensions.

- **Suggestion:** Review the code and fix according to the explanation.

```python
def main(yaml_config: str):
    dist = Distributed.get_instance()
    if fme.using_gpu():
        torch.backends.cudnn.benchmark = True
    with open(yaml_config, "r") as f:
        data = yaml.safe_load(f)
    train_config: TrainConfig = dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )

    if not os.path.isdir(train_config.experiment_dir):
        os.makedirs(train_config.experiment_dir)
    with open(os.path.join(train_config.experiment_dir, "config.yaml"), "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    train_config.configure_logging(log_filename="out.log")
    env_vars = logging_utils.retrieve_env_vars()
    gcs_utils.authenticate()
    logging_utils.log_versions()
    beaker_url = logging_utils.log_beaker_url()
    train_config.configure_wandb(env_vars=env_vars, resume=True, notes=beaker_url)
    trainer = Trainer(train_config)
    trainer.train()
    logging.info("DONE ---- rank %d" % dist.rank)
```

### ai-training

**pt-004** (critical, 95% confidence)

- **Repo:** YukiBear426__AEFIN (economics)
- **Location:** method `_train` (line 731)
- **Paper:** Non-Stationary Time Series Forecasting Based on Fourier Analysis and Cross Attention Mechanism
- **Authors:** Yuqi Xiong, Yang Wen
- **Issue:** pt-004: Issue detected
- **Explanation:** Missing optimizer.zero_grad(): Gradients accumulate across batches instead of being reset. This causes incorrect updates, exploding gradients, and NaN losses. Call optimizer.zero_grad() before each backward() in your training loop.

- **Suggestion:** Review the code and fix according to the explanation.

```python
def _sep_train(self):
        with torch.enable_grad(), tqdm(total=self.train_steps,position=0, leave=True) as progress_bar:
            self.model.train()
                                  
                                        
            times = []
            train_loss = []
            for i, (
                batch_x,
                batch_y,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in enumerate(self.train_loader):
                origin_y = origin_y.to(self.device)
                self.n_model_optim.zero_grad()
                self.f_model_optim.zero_grad()
                bs = batch_x.size(0)
                batch_x = batch_x.to(self.device, dtype=torch.float32)
                batch_y = batch_y.to(self.device, dtype=torch.float32)
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                start = time.time()

                pred, true = self._process_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )

                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = origin_y
                
                loss = self.loss_func(pred, true)
                lossn = self.model.nm.loss(true)

                
                loss.backward(retain_graph=True)
                lossn.backward(retain_graph=True)

                                                 
                                                                 
                   
                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.f_model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )

                self.n_model_optim.step()
                self.f_model_optim.step()
                
                
                end = time.time()
                times.append(end-start)
                
            print("average iter: {}ms", np.mean(times)*1000)
                
            return train_loss
```

**ml-010** (critical, 95% confidence)

- **Repo:** songtaoliu0823__crebm (chemistry)
- **Location:** function `main` (line 90)
- **Paper:** Preference Optimization for Molecule Synthesis with Conditional Residual Energy-based Models
- **Authors:** Songtao Liu et al.
- **Issue:** ml-010: Issue detected
- **Explanation:** Multi-test leakage: No held-out test set. Validation set used for both tuning and final evaluation. Create a separate test set that is never used during model development.

- **Suggestion:** Review the code and fix according to the explanation.

```python
train_data = TensorDataset(train_fps, train_costs)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)


val_data = TensorDataset(val_fps, val_costs)
val_sampler = RandomSampler(val_data)
```

**pt-009** (high, 95% confidence)

- **Repo:** PaddlePaddle__PaddleScience (earth_science)
- **Location:** function `util.train_one_epoch` (line 78)
- **Paper:** GenCast: Diffusion-based ensemble forecasting for medium-range weather
- **Authors:** Ilan Price et al.
- **Issue:** pt-009: Issue detected
- **Explanation:** Loss tensor accumulation: Accumulating loss tensors directly keeps entire computation graphs in memory, causing memory leaks. Extract the scalar value using loss.item() or loss.detach().cpu().item() before accumulating.

- **Suggestion:** Review the code and fix according to the explanation.

```python
total_epochs = config["num_epochs"] + 1
    while epoch < total_epochs:
        util.train_one_epoch(
            epoch,
            model,
            trainloader,
            optimizer,
```

### scientific-numerical

**num-005** (high, 95% confidence)

- **Repo:** ArnaudFerre__C-Norm (biology)
- **Location:** function `normalizeEmbedding` (line 49)
- **Paper:** C-Norm: a neural approach to few-shot entity normalization
- **Authors:** Arnaud Ferré et al.
- **Issue:** num-005: Issue detected
- **Explanation:** Division by zero when normalization factor (std, range, or norm) is zero for constant/zero-valued data, producing inf or NaN. Add an epsilon guard or explicit zero check before dividing.


- **Suggestion:** Review the code and fix according to the explanation.

```python
def normalizeEmbedding(vst_onlyTokens):

    for token in vst_onlyTokens.keys():
        vst_onlyTokens[token] = vst_onlyTokens[token] / numpy.linalg.norm(vst_onlyTokens[token])

    return vst_onlyTokens
```

**py-001** (medium, 95% confidence)

- **Repo:** ArnaudFerre__C-Norm (biology)
- **Location:** function `SCNN` (line 180)
- **Paper:** C-Norm: a neural approach to few-shot entity normalization
- **Authors:** Arnaud Ferré et al.
- **Issue:** py-001: Issue detected
- **Explanation:** Mutable default argument: the default object is shared across all calls. Use None as default and create inside the function.


- **Suggestion:** Review the code and fix according to the explanation.

```python
def SCNN(vst_onlyTokens, dl_terms, dl_associations, vso,
         nbEpochs=150, batchSize=64,
         l_numberOfFilters=[4000], l_filterSizes=[1],
         phraseMaxSize=15):

    data, labels, l_unkownTokens, l_uncompleteExpressions = prepare2D_data(vst_onlyTokens, dl_terms, dl_associations, vso, phraseMaxSize)

    embeddingSize = data.shape[2]
    ontoSpaceSize = labels.shape[2]

    inputLayer = Input(shape=(phraseMaxSize, embeddingSize))

    l_subLayers = list()
    for i, filterSize in enumerate(l_filterSizes):

        convLayer = (layers.Conv1D(l_numberOfFilters[i], filterSize, strides=1, kernel_initializer=initializers.GlorotUniform()))(inputLayer)

        outputSize = phraseMaxSize - filterSize + 1
        pool = (layers.MaxPool1D(pool_size=outputSize))(convLayer)

        activationLayer = (layers.LeakyReLU(alpha=0.3))(pool)

        l_subLayers.append(activationLayer)

    if len(l_filterSizes) > 1:
        concatenateLayer = (layers.Concatenate(axis=-1))(l_subLayers)                                                  
    else:
        concatenateLayer = l_subLayers[0]

    convModel = Model(inputs=inputLayer, outputs=concatenateLayer)
    fullmodel = models.Sequential()
    fullmodel.add(convModel)

    fullmodel.add(layers.Dense(ontoSpaceSize, kernel_initializer=initializers.GlorotUniform()))

    fullmodel.summary()
    fullmodel.compile(optimizer=optimizers.Nadam(), loss=losses.LogCosh(), metrics=[metrics.CosineSimilarity(), metrics.MeanSquaredError()])
    fullmodel.fit(data, labels, epochs=nbEpochs, batch_size=batchSize)

    return fullmodel, vso, l_unkownTokens
```

**py-002** (medium, 95% confidence)

- **Repo:** ArnaudFerre__C-Norm (biology)
- **Location:** function `normalizeEmbedding` (line 49)
- **Paper:** C-Norm: a neural approach to few-shot entity normalization
- **Authors:** Arnaud Ferré et al.
- **Issue:** py-002: Issue detected
- **Explanation:** NumPy arrays are passed by reference. Modifying inside a function silently changes the original. Document this behavior or .copy() on entry.


- **Suggestion:** Review the code and fix according to the explanation.

```python
def normalizeEmbedding(vst_onlyTokens):

    for token in vst_onlyTokens.keys():
        vst_onlyTokens[token] = vst_onlyTokens[token] / numpy.linalg.norm(vst_onlyTokens[token])

    return vst_onlyTokens
```

### scientific-performance

**perf-001** (high, 95% confidence)

- **Repo:** ArnaudFerre__C-Norm (biology)
- **Location:** function `prepare2D_data` (line 102)
- **Paper:** C-Norm: a neural approach to few-shot entity normalization
- **Authors:** Arnaud Ferré et al.
- **Issue:** perf-001: Issue detected
- **Explanation:** Python loop over NumPy array: 100-1000x slower than vectorized operation. Use broadcasting, np.where, or fancy indexing.


- **Suggestion:** Review the code and fix according to the explanation.

```python
def prepare2D_data(vst_onlyTokens, dl_terms, dl_associations, vso, phraseMaxSize):

                                                                                
    nbTerms = len(dl_terms.keys())
    sizeVST = word2term.getSizeOfVST(vst_onlyTokens)
    sizeVSO = word2term.getSizeOfVST(vso)

    X_train = numpy.zeros((nbTerms, phraseMaxSize, sizeVST))
    Y_train = numpy.zeros((nbTerms, 1, sizeVSO))

    l_unkownTokens = list()
    l_uncompleteExpressions = list()

    for i, id_term in enumerate(dl_associations.keys()):
                                                       
                                                                                              

        for id_concept in dl_associations[id_term]:
            Y_train[i][0] = vso[id_concept]
            for j, token in enumerate(dl_terms[id_term]):
                if j < phraseMaxSize:
                    if token in vst_onlyTokens.keys():
                        X_train[i][j] = vst_onlyTokens[token]
                    else:
                        l_unkownTokens.append(token)
                else:
                    l_uncompleteExpressions.append(id_term)
            break                                                                                                
                                                                                    

    return X_train, Y_train, l_unkownTokens, l_uncompleteExpressions
```

**perf-002** (high, 95% confidence)

- **Repo:** MIRALab-USTC__DD-RetroDCVAE (chemistry)
- **Location:** function `main` (line 269)
- **Paper:** Modeling Diverse Chemical Reactions for Single-step Retrosynthesis via Discrete Latent Variables
- **Authors:** Huarui He et al.
- **Issue:** perf-002: Issue detected
- **Explanation:** Array allocation in loop: creating arrays inside loops causes repeated malloc/free operations. Pre-allocate the output array before the loop and fill it using indexing.

- **Suggestion:** Review the code and fix according to the explanation.

```python
def main(args):
    parsing.log_args(args)

                                            
    if not os.path.exists(args.vocab_file):
        raise ValueError(f"Vocab file {args.vocab_file} not found!")
    vocab = load_vocab(args.vocab_file)
    vocab_tokens = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]

                                            
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.model == "s2s":
        model_class = Seq2Seq
        dataset_class = S2SDataset
    elif args.model == "g2s_series_rel":
        model_class = Graph2SeqSeriesRel
        dataset_class = G2SDataset
        assert args.compute_graph_distance
    else:
        raise ValueError(f"Model {args.model} not supported!")

    model = model_class(args, vocab)
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            xavier_uniform_(p)

    if args.load_from:
        state = torch.load(args.load_from)
        pretrain_args = state["args"]
        pretrain_state_dict = state["state_dict"]
        logging.info(f"Loaded pretrained state_dict from {args.load_from}")
        model.load_state_dict(pretrain_state_dict, strict=False)

    model.to(device)
    model.train()

    logging.info(model)
    logging.info(f"Number of parameters = {param_count(model)}")

                                                
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    scheduler = NoamLR(
        optimizer,
        model_size=args.decoder_hidden_size,
        warmup_steps=args.warmup_steps
    )

                                           
    train_dataset = dataset_class(args, file=args.train_bin)
    valid_dataset = dataset_class(args, file=args.valid_bin)

    total_step = 0
    accum = 0
    losses, accs = [], []

                                                             
    scaler = torch.cuda.amp.GradScaler(enabled=args.enable_amp)

    o_start = time.time()

    logging.info("Start training")
    logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for epoch in range(args.epoch):
        model.zero_grad()

        train_dataset.sort()
        train_dataset.shuffle_in_bucket(bucket_size=1000)
        train_dataset.batch(
            batch_type=args.batch_type,
            batch_size=args.train_batch_size
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda _batch: _batch[0],
            num_workers=16,
            pin_memory=True
        )

        for batch_idx, batch in enumerate(train_loader):
            if total_step > args.max_steps:
                logging.info("Max steps reached, finish training")
                exit(0)

            batch.to(device)
            with torch.autograd.profiler.profile(enabled=args.do_profile,
                                                 record_shapes=args.record_shapes,
                                                 use_cuda=torch.cuda.is_available()) as prof:

                                                                         
                with torch.cuda.amp.autocast(enabled=args.enable_amp):
                    loss_all, acc = model(batch)
                    old_loss, aux_loss, kld_loss = loss_all
                    kla_coef = min(math.tanh(2. * total_step / args.max_steps - 3) + 1, 1) * 0.03
                    loss = old_loss + 0.1 * aux_loss + kla_coef * kld_loss

                                                             
                                                                                           
                scaler.scale(loss).backward()

                losses.append(loss.item())
                accs.append(acc.item() * 100)

                accum += 1

                if accum == args.accumulation_count:
                                                                                    
                    scaler.unscale_(optimizer)

                                                                                                      
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

                                                                                                       
                    scaler.step(optimizer)

                                                           
                    scaler.update()

                    scheduler.step()

                    g_norm = grad_norm(model)
                    model.zero_grad()
                    accum = 0
                    total_step += 1

            if args.do_profile:
                logging.info(prof
                             .key_averages(group_by_input_shape=args.record_shapes)
                             .table(sort_by="cuda_time_total"))
                sys.stdout.flush()

            if (accum == 0) and (total_step > 0) and (total_step % args.log_iter == 0):
                logging.info(f"Step {total_step}, loss: {np.mean(losses)}, acc: {np.mean(accs)}, "
                             f"p_norm: {param_norm(model)}, g_norm: {g_norm}, "
                             f"lr: {get_lr(optimizer): .6f}, elapsed time: {time.time() - o_start: .0f}")
                sys.stdout.flush()
                losses, accs = [], []

            if (accum == 0) and (total_step > 0) and (total_step % args.eval_iter == 0):
                model.eval()
                eval_count = 100
                eval_meters = [0.0, 0.0]

                valid_dataset.sort()
                valid_dataset.shuffle_in_bucket(bucket_size=1000)
                valid_dataset.batch(
                    batch_type=args.batch_type,
                    batch_size=args.valid_batch_size
                )
                valid_loader = DataLoader(
                    dataset=valid_dataset,
                    batch_size=1,
                    shuffle=True,
                    collate_fn=lambda _batch: _batch[0],
                    num_workers=16,
                    pin_memory=True
                )

                with torch.no_grad():
                    for eval_idx, eval_batch in enumerate(valid_loader):
                        if eval_idx >= eval_count:
                            break
                        eval_batch.to(device)

                        eval_loss_all, eval_acc = model(eval_batch)
                        old_loss, aux_loss, kld_loss = eval_loss_all
                        kla_coef = min(math.tanh(2. * total_step / args.max_steps - 3) + 1, 1) * 0.03
                        eval_loss = old_loss + 0.1 * aux_loss + kla_coef * kld_loss
                        eval_meters[0] += eval_loss.item() / eval_count
                        eval_meters[1] += eval_acc * 100 / eval_count

                logging.info(f"Evaluation (with teacher) at step {total_step}, eval loss: {eval_meters[0]}, "
                             f"eval acc: {eval_meters[1]}")
                logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                sys.stdout.flush()

                model.train()

            if (accum == 0) and (total_step > 0) and (total_step % args.save_iter == 0):
                n_iter = total_step // args.save_iter - 1

                model.eval()
                eval_count = 100

                valid_dataset.sort()
                valid_dataset.shuffle_in_bucket(bucket_size=1000)
                valid_dataset.batch(
                    batch_type=args.batch_type,
                    batch_size=args.valid_batch_size
                )
                valid_loader = DataLoader(
                    dataset=valid_dataset,
                    batch_size=1,
                    shuffle=True,
                    collate_fn=lambda _batch: _batch[0],
                    num_workers=16,
                    pin_memory=True
                )

                accs_token = []
                accs_seq = []

                with torch.no_grad():
                    for eval_idx, eval_batch in enumerate(valid_loader):
                        if eval_idx >= eval_count:
                            break

                        eval_batch.to(device)
                        results = model.predict_step(
                            reaction_batch=eval_batch,
                            batch_size=eval_batch.size,
                            beam_size=args.beam_size,
                            n_best=1,
                            temperature=1.0,
                            min_length=args.predict_min_len,
                            max_length=args.predict_max_len
                        )[0]
                        predictions = [t[0].cpu().numpy() for t in results["predictions"]]

                        for i, prediction in enumerate(predictions):
                            acc_seq, acc_token = 0, 0
                            target_ids = None
                            for j in torch.arange(eval_batch.pres[i], eval_batch.posts[i]):
                                tgt_length = valid_dataset.tgt_lens[j]
                                tgt_token_ids = valid_dataset.tgt_token_ids[j][:tgt_length]
                                acc_seq = max(np.array_equal(tgt_token_ids, prediction[:tgt_length]), acc_seq)
                                while len(prediction) < tgt_length:
                                    prediction = np.append(prediction, vocab["_PAD"])
                                if np.mean(tgt_token_ids == prediction[:tgt_length])>=acc_token:
                                    acc_token = np.mean(tgt_token_ids == prediction[:tgt_length])
                                    target_ids = tgt_token_ids

                            accs_token.append(acc_token)
                            accs_seq.append(acc_seq)

                            if eval_idx % 20 == 0 and i == 0:
                                logging.info(f"Target text: {' '.join([vocab_tokens[idx] for idx in target_ids])}")
                                logging.info(f"Predicted text: {' '.join([vocab_tokens[idx] for idx in prediction])}")
                                logging.info(f"acc_token: {acc_token}, acc_seq: {acc_seq}\n")

                logging.info(f"Evaluation (without teacher) at step {total_step}, "
                             f"eval acc (token): {np.mean(accs_token)}, "
                             f"eval acc (sequence): {np.mean(accs_seq)}")
                logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                sys.stdout.flush()

                model.train()

                logging.info(f"Saving at step {total_step}")
                sys.stdout.flush()

                state = {
                    "args": args,
                    "state_dict": model.state_dict()
                }
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)

                torch.save(state, os.path.join(args.save_dir, f"model.{total_step}_{n_iter}.pt"))
                if n_iter >= args.keep_last_ckpt-1:
                    old_iter = n_iter - args.keep_last_ckpt
                    old_path = os.path.join(args.save_dir, f"model_{(old_iter+1) * args.save_iter}_{old_iter}.pt")
                    shutil.rmtree(old_path)

                
        if (args.accumulation_count > 1) and (accum > 0):
            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            model.zero_grad()
            accum = 0
```

**perf-002** (high, 95% confidence)

- **Repo:** MLRG-CEFET-RJ__stconvs2s (earth_science)
- **Location:** function `run_arima` (line 74)
- **Paper:** STConvS2S: Spatiotemporal Convolutional Sequence to Sequence Network for Weather Forecasting
- **Authors:** Rafaela Castro et al.
- **Issue:** perf-002: Issue detected
- **Explanation:** Array allocation in loop: creating arrays inside loops causes repeated malloc/free operations. Pre-allocate the output array before the loop and fill it using indexing.

- **Suggestion:** Review the code and fix according to the explanation.

```python
def run_arima(df, chirps, step):
    series = None
    rmse_val, mae_val = 0.,0. 
    rmse_mean, mae_mean = -999., -999.
    lat = df['lat'].unique()
    lon = df['lon'].unique()
    try:
        series = df['precip'] if (chirps) else df['air_temp']
        if ((series > 0).any()):
            split = len(series) - (step + 5)
            train = series[:split].values
            test = series[split:].values
            test_sequence = create_test_sequence(test, step)
            for observation, sequence in zip(test,test_sequence):
                start_index = len(train)
                end_index = start_index + (step-1)
                model = SARIMAX(train, order=(5,0,1)) 
                results = model.fit(disp=False) 
                pred_sequence = results.predict(start=start_index, end=end_index, dynamic=False)
                rmse_val += rmse(sequence, pred_sequence) 
                mae_val += mean_absolute_error(sequence, pred_sequence)
                np.append(train,observation) 
            
            rmse_mean = rmse_val/len(test_sequence) 
            mae_mean = mae_val/len(test_sequence)
            print(f'\n=> Model ARIMA lat: {lat}, lon: {lon}')
            print(f'RMSE: {rmse_mean:.8f}')
            print(f'MAE: {mae_mean:.8f}')    
        else:
            print(f'\n** lat: {lat}, lon: {lon} has all zero values')
    except Exception as e:
        print(f'\n## lat: {lat}, lon: {lon} error: {e}')
    
    sys.stdout.flush()
    return (rmse_mean, mae_mean)
```

### scientific-reproducibility

**rep-002** (high, 95% confidence)

- **Repo:** zhixunlee__fairgb (social_science)
- **Location:** function `run` (line 40)
- **Paper:** Rethinking Fair Graph Neural Networks from Re-balancing
- **Authors:** ZHIXUN LI et al.
- **Issue:** rep-002: Issue detected
- **Explanation:** CUDA non-determinism: some GPU ops are non-deterministic by default. Set torch.use_deterministic_algorithms(True) and torch.backends.cudnn.benchmark = False.


- **Suggestion:** Review the code and fix according to the explanation.

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

**rep-002** (high, 95% confidence)

- **Repo:** YukiBear426__AEFIN (economics)
- **Location:** function `run` (line 958)
- **Paper:** Non-Stationary Time Series Forecasting Based on Fourier Analysis and Cross Attention Mechanism
- **Authors:** Yuqi Xiong, Yang Wen
- **Issue:** rep-002: Issue detected
- **Explanation:** CUDA non-determinism: some GPU ops are non-deterministic by default. Set torch.use_deterministic_algorithms(True) and torch.backends.cudnn.benchmark = False.


- **Suggestion:** Review the code and fix according to the explanation.

```python
def run(self, seed=42) -> Dict[str, float]:
        if hasattr(self, "finished") and self.finished is True:
            print("Experiment finished!!!")
            return {}
```

**rep-006** (high, 95% confidence)

- **Repo:** YukiBear426__AEFIN (economics)
- **Location:** function `_init_data_loader` (line 254)
- **Paper:** Non-Stationary Time Series Forecasting Based on Fourier Analysis and Cross Attention Mechanism
- **Authors:** Yuqi Xiong, Yang Wen
- **Issue:** rep-006: Issue detected
- **Explanation:** DataLoader workers inherit identical random states. Add worker_init_fn to seed each worker uniquely for proper randomization.

- **Suggestion:** Review the code and fix according to the explanation.

```python
return str(ident_md5)

    def _init_data_loader(self):
        self.dataset : TimeSeriesDataset = self._parse_type(self.dataset_type)(root=self.data_path)
        self.scaler = self._parse_type(self.scaler_type)()
        self.dataloader = ChunkSequenceTimefeatureDataLoader(
```

---

*Analysis conducted: 2026-03-17 | Report generated: 2026-03-17 20:58*
