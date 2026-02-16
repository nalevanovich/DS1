#visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def plot_target_balance(df):
    print("\n--- Проверяем баланс Target ---")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', hue='target', data=df, palette='pastel', legend=False)
    plt.title('Распределение Target')
    plt.xlabel('Целевая переменная (0 - не ищет, 1 - ищет)')
    plt.ylabel('Количество')
    plt.show()

def plot_gender_distribution(df):
    fig = plt.figure(figsize=(15,6))

    ax0 = fig.add_subplot(1, 2, 1)
    sns.countplot(x="gender", hue="gender", data=df, ax=ax0, palette='pastel', legend=False)
    ax0.set_title("Общее распределение по полу")
    ax0.set_ylabel("Количество")

    ax1 = fig.add_subplot(1, 2, 2)
    sns.countplot(x="gender", hue="target", data=df, ax=ax1, palette='pastel', zorder=3)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, ["Not Looking", "Looking"], title="Target")
    ax1.set_title("Распределение по полу и таргету")
    ax1.set_ylabel("Количество")

    plt.show()

def plot_city_development(df):
    fig = plt.figure(figsize=(15, 6))
    
    ax0 = fig.add_subplot(1, 2, 1)
    sns.kdeplot(df["city_development_index"], ax=ax0, fill=True, color='skyblue', zorder=3)
    ax0.set_title("Индекс городского развития ")

    ax1 = fig.add_subplot(1, 2, 2)
    sns.kdeplot(df.loc[(df["target"]==0), "city_development_index"], ax=ax1, label="Not Looking", fill=True, color='red', alpha=0.3)
    sns.kdeplot(df.loc[(df["target"]==1), "city_development_index"], ax=ax1, label="Looking", fill=True, color='blue', alpha=0.3)
    ax1.set_title("Индекс городского развития по таргету")
    ax1.legend()

    plt.show()

def plot_training_hours(df):
    fig = plt.figure(figsize=(15, 6))
    
    ax0 = fig.add_subplot(1, 2, 1)
    sns.kdeplot(df["training_hours"], ax=ax0, fill=True, color='salmon', zorder=3)
    ax0.set_title("Распределение часов обучения")

    ax1 = fig.add_subplot(1, 2, 2)
    sns.kdeplot(df.loc[(df["target"]==0), "training_hours"], ax=ax1, label="Not Looking", fill=True, color='red', alpha=0.3)
    sns.kdeplot(df.loc[(df["target"]==1), "training_hours"], ax=ax1, label="Looking", fill=True, color='blue', alpha=0.3)
    ax1.set_title("Часы обучения по таргету")
    ax1.legend()
    
    plt.show()

def plot_major_discipline(df):
    fig = plt.figure(figsize=(15, 6))
    
    ax0 = fig.add_subplot(1, 2, 1)
    sns.countplot(x=df["major_discipline"], hue="major_discipline", data=df, ax=ax0, palette='pastel', legend=False)
    ax0.tick_params(axis='x', rotation=45)
    ax0.set_title("Количество по дисциплинам")
    ax0.set_ylabel("Количество")
    
    ax1 = fig.add_subplot(1, 2, 2)
    sns.countplot(x="major_discipline", hue="target", data=df, ax=ax1, palette='pastel', zorder=3)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, ["Not Looking", "Looking"], title="Target")
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_title("Дисциплины по таргету")
    ax1.set_ylabel("Количество")
    
    plt.show()

def plot_education_level(df):
    plt.figure(figsize=(10, 8))
    counts = df['education_level'].value_counts()
    plt.pie(counts, 
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette('pastel'),
            pctdistance=0.9,
            textprops={'fontsize': 8})
    plt.legend(counts.index, 
               title="Уровень образования",
               loc="center left", 
               bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title('Распределение по уровню образования', fontsize=16)           
    plt.show()