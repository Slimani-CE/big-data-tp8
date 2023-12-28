# TP1 : Spark MLib ✨
Ce TP vise à démontrer l'utilisation d'Apache Spark pour la réalisation de tâches d'analyse de données. Le projet contient deux classes principales :

- `LinearRegression`: Effectue une régression linéaire sur un ensemble de données publicitaires.
- `KmeansApp`: Effectue un clustering K-means sur un ensemble de données clients.

## Dépendences
![Spark Core](https://img.shields.io/badge/spark%20core-3.4.2-1572B6?style=for-the-badge&logo=apachespark&logoColor=white)
![Spark Core](https://img.shields.io/badge/spark%20SQL-3.4.2-15726?style=for-the-badge&logo=apachespark&logoColor=white)
![Spark Core](https://img.shields.io/badge/spark%20mlib-3.4.2-E5EEEE?style=for-the-badge&logo=apachespark&logoColor=white)

## LinearRegression
```java
public class Regression {
    public static void main(String[] args) {
        // Initialisation de la session Spark
        SparkSession ss = SparkSession.builder().appName("Linear Regression App").master("local[*]").getOrCreate();

        // Chargement du jeu de données depuis un fichier CSV
        Dataset<Row> dataset = ss
                .read()
                .option("inferSchema", true)
                .option("header", true)
                .csv("datasets/Advertising.csv");

        // Assemblage des colonnes en une seule colonne de features
        VectorAssembler assembler = new VectorAssembler().setInputCols(
                new String[]{"TV", "Radio", "Newspaper"}
        ).setOutputCol("Features");
        Dataset<Row> assembledDS = assembler.transform(dataset);

        // Division du jeu de données en ensembles d'entraînement et de test
        Dataset<Row>[] splits = assembledDS.randomSplit(new double[]{0.8, 0.2}, 42);
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        // Définition du modèle de régression linéaire et entraînement du modèle
        LinearRegression regression = new LinearRegression().setLabelCol("Sales").setFeaturesCol("Features");
        LinearRegressionModel model = regression.fit(train);
        Dataset<Row> predictions = model.transform(test);

        // Prédiction sur l'ensemble de test et affichage des prédictions
        System.out.println("PREDICTIONS -------------------");
        predictions.show();

        // Évaluation du modèle
        double evaluate = model.evaluate(test).r2();
        System.out.println("EVALUATION  -------------------");
        System.out.println(evaluate);
    }
}
```

## KmeansApp
```java
public class KmeansApp {
    public static void main(String[] args) {
        // Initialisation de la session Spark
        SparkSession ss = SparkSession.builder().appName("Application de K-means").master("local[*]").getOrCreate();
    
        // Chargement du jeu de données depuis un fichier CSV
        Dataset<Row> dataset = ss
                .read()
                .option("inferSchema", true)
                .option("header", true)
                .csv("datasets/Mall_Customers.csv");
    
        // Assemblage des colonnes en une seule colonne de features et normalisation
        VectorAssembler assembler = new VectorAssembler().setInputCols(new String[]{
                "CustomerID","Age","Annual Income (k$)","Spending Score (1-100)"
        }).setOutputCol("Features");
        Dataset<Row> assembledDataset = assembler.transform(dataset);
        MinMaxScaler scaler = new MinMaxScaler().setInputCol("Features").setOutputCol("NormalizedFeatures");
        Dataset<Row> normalizedDataset = scaler.fit(assembledDataset).transform(assembledDataset);
    
        // Définition du modèle K-means, entraînement du modèle et prédiction
        KMeans kMeans = new KMeans().setK(3).setFeaturesCol("NormalizedFeatures").setPredictionCol("Cluster");
        KMeansModel model = kMeans.fit(normalizedDataset);
        Dataset<Row> predictions = model.transform(normalizedDataset);
        predictions.show(200);
    }
}
```