import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_path_data = r"train.csv"
data = pd.read_csv(file_path_data)
numeric_features = ['elapsed_time', 'room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration']

for feature in numeric_features:
    mean = data[feature].mean()
    median = data[feature].median()
    mode = data[feature].mode().iloc[0]
    std = data[feature].std()

    print(f"{feature} statistics:")
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")
    print(f"Standard Deviation: {std}")
    print("\n")

    plt.figure(figsize=(10, 5))
    sns.histplot(data=data, x=feature, kde=True, bins=30)
    plt.axvline(mean, color='r', linestyle='--', label=f"Mean: {mean:.2f}")
    plt.axvline(median, color='g', linestyle='-', label=f"Median: {median:.2f}")
    plt.axvline(mode, color='b', linestyle='-', label=f"Mode: {mode:.2f}")
    plt.legend()
    plt.title(f'Histogram for {feature}')
    plt.show()

categorical_features = ['event_name', 'name', 'level', 'page', 'fullscreen', 'hq', 'music', 'level_group']

for feature in categorical_features:
    print(f"{feature} frequencies and percentages:")

    counts = data[feature].value_counts()
    percentages = data[feature].value_counts(normalize=True) * 100

    result = pd.concat([counts, percentages], axis=1, keys=['Frequency', 'Percentage'])
    print(result)
    print("\n")
    plt.figure(figsize=(10, 5))
    sns.countplot(data=data, x=feature, order=data[feature].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f'Bar plot for {feature}')
    plt.show()



# event_name  level
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='event_name', hue='level')
plt.xticks(rotation=45)
plt.title('Event Name vs Level')
plt.show()

# event_name  level_group
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='event_name', hue='level_group')
plt.xticks(rotation=45)
plt.title('Event Name vs Level Group')
plt.show()

# name  level
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='name', hue='level')
plt.xticks(rotation=45)
plt.title('Name vs Level')
plt.show()

# hq  music
plt.figure(figsize=(6, 6))
sns.countplot(data=data, x='hq', hue='music')
plt.title('HQ vs Music')
plt.show()


# elapsed_time  event_name
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='event_name', y='elapsed_time')
plt.xticks(rotation=45)
plt.title('Elapsed Time vs Event Name')
plt.show()

# elapsed_time  level
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='level', y='elapsed_time')
plt.title('Elapsed Time vs Level')
plt.show()

# hover_duration  event_name
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='event_name', y='hover_duration')
plt.xticks(rotation=45)
plt.title('Hover Duration vs Event Name')
plt.show()

# hover_duration  level
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='level', y='hover_duration')
plt.title('Hover Duration vs Level')
plt.show()

