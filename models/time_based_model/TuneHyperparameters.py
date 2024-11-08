import optuna

def objective(trial):
    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 128, 1024)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    max_lr = trial.suggest_float('max_lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)

    # Initialize model
    model = ModelBuilder(input_size, hidden_size, output_size, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    total_steps = args.num_epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)
    criterion = nn.MSELoss()
    evaluator = Evaluator(model, criterion, device)
    trainer = Trainer(model, optimizer, scheduler, criterion, device, args.patience, evaluator)

    # Train the model
    trainer.train(args.num_epochs, train_loader, val_loader)

    # Evaluate on validation set
    val_loss = evaluator.evaluate_metrics(val_loader)
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)
