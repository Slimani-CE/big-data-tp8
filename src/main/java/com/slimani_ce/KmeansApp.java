package com.slimani_ce;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

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