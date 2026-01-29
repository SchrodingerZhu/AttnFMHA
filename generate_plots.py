
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

def plot_study(study_name, misses_static, tflops_static, misses_tile, tflops_tile):
    labels = ['Top-Down', 'Tile-Based']  
    # "Static" in original code seems to refer to Top-Down based on context of previous knowledge or inference, 
    # but the labels in code were ['Static', 'Tile']. User said "Port attentionfmha variants... Keep only full_static, full_static_alt, tile, tile_alt".
    # In the plots, the x-axis groups are "Static" and "Tile".
    # The bars are "Regular" and "Alternating".
    
    # Legend for bars
    bar_labels = ['Regular (Cyclic)', 'Alternating (Sawtooth)']
    
    # Data prep
    misses_reg = [misses_static[0], misses_tile[0]]
    misses_alt = [misses_static[1], misses_tile[1]]
    
    tflops_reg = [tflops_static[0], tflops_tile[0]]
    tflops_alt = [tflops_static[1], tflops_tile[1]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Plot Misses
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, misses_reg, width, label=bar_labels[0])
    rects2 = ax.bar(x + width/2, misses_alt, width, label=bar_labels[1])
    
    ax.set_ylabel('L2 Miss Count')
    ax.set_title(f'L2 Cache Misses: Regular vs Alternating ({study_name})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    def autolabel_misses(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height/1e6:.1f}M',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel_misses(rects1)
    autolabel_misses(rects2)
    plt.tight_layout()
    plt.savefig(f'plots/{study_name.lower().replace(" ", "_")}_misses.png')
    plt.close()

    # Plot TFLOPS
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, tflops_reg, width, label=bar_labels[0])
    rects2 = ax.bar(x + width/2, tflops_alt, width, label=bar_labels[1])
    
    ax.set_ylabel('Throughput (TFLOPS)')
    ax.set_title(f'Kernel Throughput: Regular vs Alternating ({study_name})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    
    def autolabel_tflops(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    autolabel_tflops(rects1)
    autolabel_tflops(rects2)
    plt.tight_layout()
    plt.savefig(f'plots/{study_name.lower().replace(" ", "_")}_throughput.png')
    plt.close()

# Study 5 Data
# misses_static = [372244480, 122117820]
# tflops_static = [61.58, 69.34]
# misses_tile = [369644856, 127651952]
# tflops_tile = [60.74, 69.24]
plot_study('Study 5', 
           [372244480, 122117820], [61.58, 69.34],
           [369644856, 127651952], [60.74, 69.24])

# Study 6 Data
# misses_static = [629425276, 95344684]
# tflops_static = [32.21, 59.89]
# misses_tile = [350626172, 162876760]
# tflops_tile = [41.05, 66.11]
plot_study('Study 6',
           [629425276, 95344684], [32.21, 59.89],
           [350626172, 162876760], [41.05, 66.11])
