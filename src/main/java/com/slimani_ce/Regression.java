package com.slimani_ce;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

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