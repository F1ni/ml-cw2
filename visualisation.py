from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def tsne(all_features_normalized, cluster_assignments, final_typical_images, total_budget):
    features_to_plot = all_features_normalized.numpy()


    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(features_to_plot)
    print("t-SNE complete!")

    plt.figure(figsize=(10, 8))


    plt.scatter(
        features_2d[:, 0], 
        features_2d[:, 1], 
        c=cluster_assignments, 
        cmap='tab10', 
        alpha=0.3, 
        s=15       
    )


    typical_x = features_2d[final_typical_images, 0]
    typical_y = features_2d[final_typical_images, 1]

    plt.scatter(
        typical_x, 
        typical_y, 
        c='red', 
        marker='*', 
        s=350, 
        edgecolor='black',
        label=f'Typical Images (Budget={total_budget})'
    )


    plt.title("t-SNE Representation of TypiClust Clusters", fontsize=14, fontweight='bold')
    plt.legend()
    plt.axis('off') 


    plt.savefig('tsne_typiclust.png', dpi=300, bbox_inches='tight')
    plt.show()


#final_typical_images = [42076, 26315, 24916, 32635, 12071, 41009, 26783]   
def layout(final_typical_images, clean_dataset):
    fig, axes = plt.subplots(3, 10, figsize=(15, 3))
    axes = axes.flatten()

    for idx, image_id in enumerate(final_typical_images):
    
        img_tensor, true_label_id = clean_dataset[image_id]
        
    
        class_name = clean_dataset.classes[true_label_id]
        
    
        img_display = np.transpose(img_tensor.numpy(), (1, 2, 0))
        
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img_display = std * img_display + mean

        img_display = np.clip(img_display, 0, 1)
    
        axes[idx].imshow(img_display)
        axes[idx].set_title(f"ID: {image_id}\nClass: {class_name}")
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()