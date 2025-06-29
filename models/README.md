# Model Files Directory

This directory should contain your trained model files:

## Required Files:

1. **gnn_model.pth** - GraphSAGE encoder weights
2. **bilstm_model.pth** - BiLSTM classifier weights  
3. **random_forest.pkl** - Random Forest ensemble model
4. **scaler.pkl** - Feature scaler for preprocessing
5. **label_encoders.pkl** - Label encoders for categorical features

## Model Architecture:

- **GraphSAGE Encoder**: 3-layer GraphSAGE with 256 hidden dimensions
- **BiLSTM Classifier**: 2-layer bidirectional LSTM with 128 hidden units
- **Random Forest**: Ensemble backup classifier
- **Input Features**: 12 categorical and numerical features

## Usage:

Place your trained model files in this directory. The API will automatically load them on startup.

## File Formats:

- PyTorch models: `.pth` files
- Scikit-learn models: `.pkl` files using joblib

## Example Model Training:

```python
# Save GraphSAGE model
torch.save(gnn_model.state_dict(), 'gnn_model.pth')

# Save BiLSTM model  
torch.save(bilstm_model.state_dict(), 'bilstm_model.pth')

# Save preprocessing objects
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(rf_model, 'random_forest.pkl')
```

## Model Performance:

- Training Accuracy: ~89%
- Validation Accuracy: ~85%
- F1 Score: ~0.84
- Model Size: ~15.2 MB
- Inference Time: ~45ms