import argparse
import torch
from data.data import set_seed, get_data_dir, load_metadata, get_dataloaders
from train.train_cnn import get_model_by_name, train_model
from eval.evaluation_cnn import evaluate_model
# Note: for RF — only evaluate RF; assume another script handles RF training + saving
from eval.evaluation_rf import evaluate_rf_model  # renamed function-style

def train_one(model_name, args, dataloaders):
    device = args.device

    # pick augmentation mode
    if args.strong_aug:
        active_train_loader = dataloaders['train_strong']
        print("Using HEAVY augmentation")
    else:
        active_train_loader = dataloaders['train']
        print("Using LIGHT augmentation")

    model = get_model_by_name(
        model_name,
        pretrained=args.pretrained,
        dropout=args.dropout,
        num_classes=2
    ).to(device)

    print(f"Training model {model_name} ...")
    history = train_model(
        model,
        active_train_loader,
        dataloaders['val'],
        aug=args.strong_aug,
        num_epochs=args.epochs,
        lr=args.lr
    )

    if args.evaluate_after_train:
        print(f"Evaluating model {model_name} ...")
        results = evaluate_model(
            model,
            dataloaders['test'],
            device=device,
            class_names=['Non-Toxic', 'Toxic'],
            model_name=model_name,
            compute_probs=True
        )
        return model, history, results
    else:
        return model, history, None

def eval_only(model_path, args, dataloaders):
    device = args.device
    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=device)  # or use appropriate load logic
    results = evaluate_model(
        model,
        dataloaders['test'],
        device=device,
        class_names=['Non-Toxic', 'Toxic'],
        model_name=model_path,
        compute_probs=True
    )
    return model, None, results

def main():
    parser = argparse.ArgumentParser(description="Train / Evaluate CNN models")
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'all'], default='train',
                        help="train: train (and optionally eval) specified model; "
                             "eval: evaluate a pre-trained model; "
                             "all: train + eval all model variants")
    parser.add_argument('--model', type=str, choices=['b0','b0_cbam','b2_cbam'],
                        help="Model variant to run (required for mode=train or mode=all)")
    parser.add_argument('--pretrained', action='store_true', help="Use pretrained weights")
    parser.add_argument('--strong_aug', action='store_true',
                        help="Train model with strong data augmentation")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate for model head")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_path', type=str,
                        help="Path to saved model file (for mode=eval)")
    parser.add_argument('--evaluate_after_train', action='store_true',
                        help="If set during training, also run evaluation on test data after training")

    args = parser.parse_args()
    set_seed(42)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = get_data_dir()
    meta_df = load_metadata(data_dir)
    train_loader, train_strong_loader, val_loader, test_loader = get_dataloaders(
        meta_df, data_dir, batch_size=args.batch_size if hasattr(args, 'batch_size') else 32
    )
    dataloaders = {
        'train': train_loader,
        'train_strong': train_strong_loader,
        'val': val_loader,
        'test': test_loader
    }

    if args.mode == 'train':
        assert args.model, "--model is required with mode=train"
        model, history, results = train_one(args.model, args, dataloaders)
        # optionally: save model here
    elif args.mode == 'eval':
        assert args.model_path, "--model_path is required with mode=eval"
        model, history, results = eval_only(args.model_path, args, dataloaders)
    elif args.mode == 'all':
        for m in ['b0','b0_cbam','b2_cbam']:
            print("\n" + "="*30)
            print(f"Processing model: {m}")
            model, history, results = train_one(m, args, dataloaders)
            # optionally save each model
    else:
        parser.error("Unknown mode")

if __name__ == "__main__":
    main()
