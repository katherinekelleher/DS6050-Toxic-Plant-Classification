import torch
import argparse
from data.data import set_seed, get_data_dir, load_metadata, get_dataloaders
from train.train_cnn import get_model_by_name  # FIXED: Import from train_cnn, not cnn_models
from train.train_cnn import train_model
from eval.evaluation_cnn import evaluate_model  # FIXED: Use actual filename

def main(args):
    # Setup
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data_dir = get_data_dir()
    meta_df = load_metadata(data_dir)
    
    # Get dataloaders
    train_loader, train_strong_loader, val_loader, test_loader = get_dataloaders(
        meta_df, 
        data_dir,
        batch_size=args.batch_size
    )
    
    # Select which training loader to use
    if args.strong_aug:
        active_train_loader = train_strong_loader
        print("Using HEAVY augmentation")
    else:
        active_train_loader = train_loader
        print("Using LIGHT augmentation")
    
    # Get model
    model = get_model_by_name(
        args.model, 
        pretrained=args.pretrained, 
        dropout=args.dropout,
        num_classes=2
    )
    model = model.to(device)
    
    # Training - REMOVED device parameter (train_model uses global device)
    history = train_model(
        model, 
        active_train_loader, 
        val_loader, 
        num_epochs=args.epochs, 
        lr=args.lr
    )
    
    # Evaluation
    class_names = ['Non-Toxic', 'Toxic']
    results = evaluate_model(
        model, 
        test_loader, 
        device=device, 
        class_names=class_names,
        model_name=args.model, 
        compute_probs=True
    )
    
    return results, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate CNN models")
    parser.add_argument("--model", type=str, choices=["b0", "b0_cbam", "b2_cbam"], 
                       default="b0_cbam")
    parser.add_argument("--pretrained", action="store_true", 
                       help="Use pretrained ImageNet weights")
    parser.add_argument("--strong_aug", action="store_true", 
                       help="Use strong augmentation (CutMix + ColorJitter)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()
    main(args)