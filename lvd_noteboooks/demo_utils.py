import matplotlib.pyplot as plt
import seaborn as sns


def color_to_topic(color):
    if color == 'red':
        return 'Nature'
    elif color == 'blue':
        return 'Career Path'
    elif color == 'green':
        return 'Free Time'
    elif color == 'purple':
        return 'Computer Science'
    else:
        return 'other'


def visualize_dataset(data, query_point):
    plt.figure(figsize=(8, 6))
    for color in data['cluster'].unique().tolist():
        cluster_data = data[data['cluster'] == color]
        plt.scatter(cluster_data['x'], cluster_data['y'], color=color, label=f'{color_to_topic(color)} Topic', s=25)

    plt.scatter(query_point['x'],
                query_point['y'],
                color='#FFAE42',
                edgecolor='#FFAE42',
                label='Query', zorder=5, s=80
    )

    plt.title('Visualization of Synthetic Dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def plot_bucket_items(data, use_colors):
    """
    Plot the number of items in each bucket, with optional color coding by cluster.

    Parameters:
    data (DataFrame): The data frame containing the 'id', 'bucket', and 'cluster' columns.
    use_colors (bool): If True, use different colors for each cluster in the visualization.
    """
    if use_colors:
        bucket_cluster_counts = data.groupby(['bucket_str', 'cluster']).size().unstack(fill_value=0)
        plot_data = bucket_cluster_counts.reset_index().melt(id_vars='bucket_str', var_name='cluster',
                                                             value_name='count')
        palette = {cluster: cluster for cluster in data['cluster'].unique()}

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=plot_data, x='bucket_str', y='count', hue='cluster', palette=palette)
    else:
        bucket_counts = data.groupby('bucket_str').size()
        plot_data = bucket_counts.reset_index(name='count')

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=plot_data, x='bucket_str', y='count')

    plt.title('Number of Items in Each Bucket')
    plt.xlabel('Bucket')
    plt.ylabel('Count')

    for p in ax.patches:
        bar_height = int(p.get_height())
        if bar_height > 0:
            ax.annotate(f'{bar_height}', (p.get_x() + p.get_width() / 2., bar_height),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')

    plt.show()


def visualize_bucket_order(data, bucket_order):
    order_labels = [str(bucket) for bucket in bucket_order]

    cluster_colors = {'blue': 'blue', 'green': 'green', 'purple': 'purple', 'red': 'red'}

    bucket_counts = data.groupby(['bucket_str', 'cluster']).size().unstack(fill_value=0)

    plt.figure(figsize=(10, 6))

    for i, bucket in enumerate(order_labels):
        for cluster, count in bucket_counts.loc[bucket].items():
            if count > 0:
                plt.bar(i, count, color=cluster_colors[cluster])
                break

    plt.xticks(range(len(order_labels)), order_labels)

    plt.title('Number of Items from Each Cluster in Ordered Buckets')
    plt.xlabel('Bucket')
    plt.ylabel('Count')

    plt.show()