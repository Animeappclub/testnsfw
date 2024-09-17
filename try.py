from helpers.file_utils import load_model, save_model

def merge_weights(base_model, other_model, merge_weight=1.0):
    """
    Merge weights from other_model into base_model. The base_model's structure is preserved.
    Only weights (bias, key, value, etc.) are merged from other_model into the corresponding layers of base_model.

    Parameters:
    base_model: The model whose structure will be preserved.
    other_model: The model from which weights will be taken to merge into base_model.
    merge_weight: The weight for merging the other_model's weights into base_model. Default is 1.0 (100% from other_model).

    Returns:
    merged_model: The resultant model after merging weights.
    """
    merged_model = base_model.copy()  # Preserve base model structure
    
    # Iterate through each layer and merge weights
    for layer in range(len(base_model['layers'])):
        base_layer = base_model['layers'][layer]
        other_layer = other_model['layers'][layer]
        
        # Merging layer weights (e.g., 'key', 'value', 'bias', etc.)
        for param in base_layer:
            if param in other_layer:  # If the parameter exists in the other_model
                # Merge the weights
                merged_model['layers'][layer][param] = (
                    (1 - merge_weight) * base_layer[param] + merge_weight * other_layer[param]
                )
    
    return merged_model

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge weights from one model into another while preserving structure")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model file")
    parser.add_argument("--other_model", type=str, required=True, help="Path to the other model file")
    parser.add_argument("--output", type=str, required=True, help="Path for the output merged model")
    parser.add_argument("--merge_weight", type=float, default=1.0, help="Weight for merging other model's weights (1.0 means 100%)")

    args = parser.parse_args()

    # Load models
    base_model = load_model(args.base_model)
    other_model = load_model(args.other_model)

    # Perform the merging operation
    merged_model = merge_weights(base_model, other_model, args.merge_weight)

    # Save the merged model
    save_model(merged_model, args.output)

if __name__ == "__main__":
    main()
