# -*- coding: utf-8 -*-
"""
Fluency evaluating script that takes
	- TextGrid
    - f0
	- csv
files and determines linear regression multipliers X so that fX = S where f is
a vector of features created from TextGrid and f0, S is fluency vector taken
from csv-file.

Possible outputs:
    - model for combined fluency estimator
    - list of models for separate fluency estimators
    - features plotted in respect to fluency ratings, combined and separates

Created on Fri Oct 4 17:27:00 2019

@author: Herkko Salonen
"""


import argparse
import csv
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sklearn.linear_model as LinearRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import statistics
import tgre
import time


# for printing histograms
bins = np.linspace(4, 20, 17)


# Calculate prosodic features regarding silent regions and vowel and consontant lengths.
def getP678(textgrid):
    # Phones are located in the third tier if speech alignment is used
    # If recognition is used they are in the fourth tier
    phones = textgrid.tiers[2]    
    vowels = ['a', 'e', 'i', 'o', 'u', 'y', 'å', 'ä', 'ö']
    vowelLengths = []
    consonantLenghts = []
    silentRegions = []
    # Flag to check if previous letter was a vowel and calculate vowel lengths
    # accordingly
    previous = False
    for phone in phones[1:len(phones) - 1]:
        if phone.text == "":
            silentRegions.append(phone.xmax - phone.xmin)
            previous = False
        elif phone.text in vowels:
            if previous == True:
                vowelLengths[len(vowelLengths) - 1] += phone.xmax - phone.xmin
            else:
                vowelLengths.append(phone.xmax - phone.xmin)
                previous = True
        else:
            consonantLenghts.append(phone.xmax - phone.xmin)
            previous = False
        
    numberSilentRegions = len(silentRegions)
    avarageSilentLength = sum(silentRegions) / len(silentRegions)
    longestSilentRegion = max(silentRegions)
    silentRegionDeviation = statistics.stdev(silentRegions)
    avarageVowelLength = sum(vowelLengths) / len(vowelLengths)
    longestVowelLength = max(vowelLengths)
    vowelLenghtDeviation = statistics.stdev(vowelLengths)
    avarageConsonantLenght = sum(consonantLenghts) / len(consonantLenghts)
    longestConsonant = max(consonantLenghts)
    consonantLenghtDeviation = statistics.stdev(consonantLenghts)
    
    return [numberSilentRegions, avarageSilentLength, longestSilentRegion, silentRegionDeviation, avarageVowelLength, longestVowelLength, vowelLenghtDeviation, avarageConsonantLenght, longestConsonant, consonantLenghtDeviation]


# Calculate prosodic features regarding sentences
# Last silence approximates the error of ASR alignment 
def getP3(textgrid):
    # Sentences are located always in the first tier
    sentences = textgrid.tiers[0]
    
    silentRegions = []
    
    for sentence in sentences[1:len(sentences) - 1]:
        if sentence.text == "":
            silentRegions.append(sentence.xmax - sentence.xmin)
            
    if (len(silentRegions) > 0):
        avarageSilentRegion = sum(silentRegions) / len(silentRegions)
    else:
        avarageSilentRegion = 0

    lastSilence = sentences[len(sentences) - 1]
    lastSilence = lastSilence.xmax - lastSilence.xmin
        
    return [avarageSilentRegion, lastSilence]


# Calculate features from fundamental frequency information
#   1. Calculate mean freq. for every word and return the variance of these means
#   2. Calculate variance of freq. inside every word and return mean of these variances
def getPF0(textgrid, f0):
    variances = []
    meanFrequencies = []
    
    
    words = textgrid.tiers[1]
    
    for word in words[1:len(words) - 1]:
        start = round(word.xmin * 100)
        end = round(word.xmax * 100) - 1
        wordf0 = f0[start:end]
        wordf0 = wordf0[wordf0 != 0]
        if len(wordf0 > 0):
            variances.append(wordf0.var())
            meanFrequencies.append(wordf0.mean())
    
    if (len(variances) > 0 ):
        meanOfVariances = np.mean(variances)
    else:
        meanOfVariances = 0
    if (len(meanFrequencies) > 9):
        varianceOfMeans = np.var(meanFrequencies)
    else:
        varianceOfMeans = 0
    
    return [meanOfVariances, varianceOfMeans]


# Calculate lexical features introduced in Bolanos et al.
#   1. Words Per Minute
#   2. Variance of sentence reading rate
def getLexicalFeatures(textgrid):
    words = []
    for word in textgrid[1]:
        if word.text != "":
            words.append(word)
            
    WPM = len(words) / (words[len(words) - 1].xmax) * 60
    
    sentences = []
    for sentence in textgrid[0]:
        if sentence.text != "":
            sentences.append(sentence)
    
    sentenceWMP = []
    for i in sentences:
        numberOfWords = i.text.count(" ") + 1
        length = i.xmax - i.xmin
        if (length == 0):
            continue
        sentenceWMP.append(numberOfWords / length * 60)
        
    WPMvariance = np.array(sentenceWMP).var()
    
    return [WPM, WPMvariance]


# Compiles all features from separate functions
def getFeatures(textgrid, f0):
    # Empty list for features
    featureList = []
    # Add features to the list
    featureList.extend(getP678(textgrid))               # 10 features
    featureList.extend(getP3(textgrid))                 # 2 features
    featureList.extend(getPF0(textgrid, f0))            # 2 features
    featureList.extend(getLexicalFeatures(textgrid))    # 2 features
    
    # Convert list to a numpy array and return
    featureArray = np.array(featureList)
    return featureArray


def getFeatureMatrix(tgFolder, f0Folder):
    txFiles = os.listdir(tgFolder)
    f0Files = os.listdir(f0Folder)
    
    featureMatrix = np.zeros((len(txFiles), 16))
    
    for i in range(round(len(txFiles))):
        tx = tgre.TextGrid.from_file(tgFolder + txFiles[i])
        f0List = []
        with open(f0Folder + f0Files[i], 'r') as csvFile:
            csvReader = csv.reader(csvFile)
            for row in csvReader:
                f0List.append(float(row[0]))
        csvFile.close()

        f0 = np.array(f0List)
        features = getFeatures(tx, f0)
        featureMatrix[i] = features
    
    return featureMatrix


# Return combined fluency assessment
def getFluency(file):
    fluencyList = []
    with open(file, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=';')
        lineCount = 0
        for row in csvReader:
            if lineCount == 0:
                lineCount += 1
                continue
            else:
                fluencyList.append( round( int(row[1]) + int(row[2]) + int(row[3]) + int(row[4]) ) )
        
    csvFile.close()        
    fluencyMatrix = np.array(fluencyList)
    return fluencyMatrix


# Returns matrix of multiple fluency assessmentss
def getMultipleF(file):
    fluencyList = []
    with open(file, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=";")
        linecount = 0
        for row in csvReader:
            if linecount == 0:
                linecount += 1
                continue
            else:
                fluencyList.append([int(row[1]), int(row[2]), int(row[3]), int(row[4])])
        
    csvFile.close()        
    fluencyMatrix = np.array(fluencyList)
    return fluencyMatrix


# Fit features to a linear regression model
def fitLinearModel(features, y):
    reg = LinearRegression.LinearRegression().fit(features, y)
    return reg
    

# Trains and returns a statistical model with first 70% of the data
#   Arguments are paths to folders which contain respected files.
#   Argument y is path to file which contains fluency approximations. 
# NOTE: train-test split is always the same!!
def train(textgrid, pitch, y):
    featureMatrix = getFeatureMatrix(textgrid, pitch)
    fluencyMatrix = getFluency(y)
    
    split = round(fluencyMatrix.size * 0.7)
    
    model = fitLinearModel(featureMatrix[0:split], fluencyMatrix[0:split])
    
    predictions = model.predict(featureMatrix[split:fluencyMatrix.size])

    mae = np.sum(np.abs(predictions - fluencyMatrix[split:fluencyMatrix.size])) / predictions.size
    print("Mean absolute error is ", mae, " for test data")
    rmse = np.sqrt(np.sum(np.square(predictions - fluencyMatrix[split:fluencyMatrix.size])) / predictions.size)
    print("Root-mean-square error is ", rmse, " for test data")
    
    # Calculate normal distributions of the predictions and plot it compared to real fluency data
    mu_p = predictions.mean()
    sigma_p = predictions.std()
    print("Prediction mean: ", mu_p, " standard deviation: ", sigma_p)
    mu_f = fluencyMatrix[split:fluencyMatrix.size].mean()
    sigma_f = fluencyMatrix[split:fluencyMatrix.size].std()
    print("Human assessment mean: ", mu_f, " standard deviation: ", sigma_f)
    
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 6)
    axs[0].set_title('Predictions')
    axs[0].hist(predictions, bins=bins)
    axs[1].set_title('Human assessment')
    axs[1].hist(fluencyMatrix[split:fluencyMatrix.size], bins=bins)
    
    return model


# Trains and returns a statistical model with first 70% of the data
#   Arguments are paths to folders which contain respected files.
#   Argument y is path to file which contains fluency approximations. 
# NOTE: train-test split is done randomly!!
def testRandom(textgrid, pitch, y):
    featureMatrix = getFeatureMatrix(textgrid, pitch)
    fluencyMatrix = getFluency(y)
    
    combinations =  []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(featureMatrix, fluencyMatrix, test_size=0.3)
        
        # Fit features to a model
        model = fitLinearModel(X_train, y_train)
        #model = LinearRegression.LogisticRegression(penalty='l2', tol=0.000001, C=0.001, class_weight='balanced', solver='lbfgs', max_iter=1000, multi_class='multinomial').fit(X_train, y_train)
        #model = KNeighborsClassifier(5, weights='distance', algorithm='auto').fit(X_train, y_train)
        #model = DecisionTreeClassifier().fit(X_train, y_train)
        #model = RandomForestClassifier(n_estimators=10000).fit(X_train, y_train)
        #model = MLPClassifier(hidden_layer_sizes=(16, 100, 100, 100, 17), max_iter=10000).fit(X_train, y_train)
        #model = AdaBoostClassifier(n_estimators=10000).fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        mae = np.sum(np.abs(predictions - y_test)) / predictions.size
        print("Mean absolute error is ", mae, " for test data")
        rmse = np.sqrt(np.sum(np.square(predictions - y_test)) / predictions.size)
        print("Root-mean-square error is ", rmse, " for test data")
        
        # Calculate normal distributions of the predictions and plot it compared to real fluency data
        mu_p = predictions.mean()
        sigma_p = predictions.std()
        print("Prediction mean: ", mu_p, " standard deviation: ", sigma_p)
        mu_f = y_test.mean()
        sigma_f = y_test.std()
        print("Human assessment mean: ", mu_f, " standard deviation: ", sigma_f)
        
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(16, 6)
        plt.xlabel('Arvio', fontsize=24)
        plt.ylabel('Frekvenssi', fontsize=24)
        
        axs[0].set_title('Järjestelmä', fontsize=24)
        axs[0].hist(predictions, bins=bins)
        axs[0].tick_params(axis='both', which='major', labelsize=18)
        axs[0].tick_params(axis='both', which='minor', labelsize=18)
        
        axs[1].set_title('Ihminen', fontsize=24)
        axs[1].hist(y_test, bins=bins)
        axs[1].tick_params(axis='both', which='major', labelsize=18)
        axs[1].tick_params(axis='both', which='minor', labelsize=18)
        
        plt.savefig("kuvat/{}.svg".format(i), format='svg')
        
        result = [mae, rmse, r2_score(y_test, predictions), mu_p, mu_f, sigma_p, sigma_f]
        combinations.append(result)
    
    with open("testitestiModel0212.csv", 'w', newline='\n') as csvfile: 
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        for i in combinations:
            wr.writerow(i)
    
    csvfile.close()
    return combinations


# Test different feature combinations for multiple fluency assessments
def testFeaturesMultiple(textgrid, pitch, y):
    featureMatrix = getFeatureMatrix(textgrid, pitch)
    fluencyMatrix = getMultipleF(y)

    # List of feature indexes:
    featureIndexes = []
    for i in range(featureMatrix.shape[1]):
        featureIndexes.append(i)

    X_train, X_test, y_train, y_test = train_test_split(featureMatrix, fluencyMatrix, test_size=0.3)
    
    # Train models with all possible feature combinations, test the model, calculate MAE and RMSE and save results
    painotus = []
    sujuvuus = []
    tahti = []
    tunneilmaisu = []
    for i in range(featureMatrix.shape[1]):
        # Create list of all combinations
        comb = list(itertools.combinations(featureIndexes, i+1))
        # Iterate through the list, train a model with selected features
        for j in range(len(comb)):
            print(comb[j])
            for k in range(fluencyMatrix.shape[1]):
                model = LinearRegression.LinearRegression().fit(X_train[:, comb[j]], y_train[:,k])
                predictions = model.predict(X_test[:, comb[j]]) 
                
                corrects = 0
                for l in range(predictions.size):
                    if round(predictions[l]) == y_test[l, k]:
                        corrects += 1
                
                mae = np.sum(np.abs(predictions - y_test[:,k]) / predictions.size)
                rmse = np.sqrt(np.sum(np.square(predictions - y_test[:,k])) / predictions.size)
                score = r2_score(y_test[:,k], predictions)
                result = [str(comb[j]), mae, rmse, score, corrects/predictions.size]
                
                # Save results to respective array
                if k == 0:
                    painotus.append(result)
                elif k == 1:
                    sujuvuus.append(result)
                elif k == 2:
                    tahti.append(result)
                elif k == 3:
                    tunneilmaisu.append(result)
    
    with open("painotus0212.csv", 'w', newline='\n') as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        for i in painotus:
            wr.writerow(i)
    csvfile.close()        

    with open("sujuvuus0212.csv", 'w', newline='\n') as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        for i in sujuvuus:
            wr.writerow(i)
    csvfile.close()        

    with open("tahti0212.csv", 'w', newline='\n') as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        for i in tahti:
            wr.writerow(i)
    csvfile.close()    

    with open("tunneilmaisu0212.csv", 'w', newline='\n') as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        for i in tunneilmaisu:
            wr.writerow(i)
    csvfile.close()    


# Test models for all fluency criteria plot test results and return models
def trainMultiple(textgrid, pitch, y):
    featureMatrix = getFeatureMatrix(textgrid, pitch)
    fluencyMatrix = getMultipleF(y)

    # List of feature indexes:
    featureIndexes = []
    for i in range(featureMatrix.shape[1]):
        featureIndexes.append(i)

    X_train_a, X_test_a, y_train, y_test = train_test_split(featureMatrix, fluencyMatrix, test_size=0.3)
    model_list = []
    allPredictions = []
    for i in range(fluencyMatrix.shape[1]):
        if i == 0:
            X_train = X_train_a[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 15]]
            X_test = X_test_a[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 15]]
        elif i == 1:
            X_train = X_train_a[:, [0, 2, 4, 5, 6, 10, 11, 12, 14]]
            X_test = X_test_a[:, [0, 2, 4, 5, 6, 10, 11, 12, 14]]            
        elif i == 2:
            X_train = X_train_a[:, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15]]
            X_test = X_test_a[:, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15]]            
        elif i  == 3:
            X_train = X_train_a[:, [0, 2, 4, 6, 7, 10, 11, 12, 13]]
            X_test = X_test_a[:, [0, 2, 4, 6, 7, 10, 11, 12, 13]]
        
        model = fitLinearModel(X_train, y_train[:, i])
        predictions = model.predict(X_test)
        
        model2 = fitLinearModel(X_train_a, y_train[:, i])
        predictions2 = model2.predict(X_test_a)
        
        rmse = np.sqrt(np.sum(np.square(predictions - y_test[:,i])) / predictions.size)
        rmse2 = np.sqrt(np.sum(np.square(predictions2 - y_test[:,i])) / predictions2.size)

        print("RMSE on test data: ")
        print( rmse )
        print("All features: ")
        print( rmse2 )
        
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(16, 6)
        axs[0].set_title('Predictions')
        axs[0].hist(predictions, bins=[1, 2, 3, 4, 5])
        axs[1].set_title('Human assessment')
        axs[1].hist(y_test[:,i], bins=[1, 2, 3, 4, 5])
        
        model_list.append(model)
        allPredictions.append([predictions, y_test[:,i]])

    painotus = allPredictions[0]
    sujuvuus = allPredictions[1]
    tahti = allPredictions[2]
    tunne = allPredictions[3]
    
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(15, 15)
    plt.xlabel('Järjestelmä', fontsize=18)
    plt.ylabel('Ihminen', fontsize=18)
    axs[0, 0].set_title('Painotus', fontsize=24)
    axs[0, 0].plot([0, 5], [0, 5], color='r')
    axs[0, 0].scatter(painotus[0], painotus[1])
    axs[0, 0].tick_params(axis='both', which='major', labelsize=18)
    axs[0, 0].tick_params(axis='both', which='minor', labelsize=18)
    
    axs[0, 1].set_title('Sujuvuus', fontsize=24)
    axs[0, 1].plot([0, 5], [0, 5], color='r')
    axs[0, 1].scatter(sujuvuus[0], sujuvuus[1])
    axs[0, 1].tick_params(axis='both', which='major', labelsize=18)
    axs[0, 1].tick_params(axis='both', which='minor', labelsize=18)
    
    axs[1, 0].set_title('Tahti', fontsize=24)
    axs[1, 0].plot([0, 5], [0, 5], color='r')
    axs[1, 0].scatter(tahti[0], tahti[1])
    axs[1, 0].tick_params(axis='both', which='major', labelsize=18)
    axs[1, 0].tick_params(axis='both', which='minor', labelsize=18)
    
    axs[1, 1].set_title('Tunneilmaisu', fontsize=24)
    axs[1, 1].plot([0, 5], [0, 5], color='r')
    axs[1, 1].scatter(tunne[0], tunne[1])
    axs[1, 1].tick_params(axis='both', which='major', labelsize=18)
    axs[1, 1].tick_params(axis='both', which='minor', labelsize=18)

    plt.savefig("kuvat/testivideoust.svg", format='svg')

    return model_list, allPredictions


# Test all feature combinations with static split and smaller training set
def featureTest(textgrid, pitch, y):
    featureMatrix = getFeatureMatrix(textgrid, pitch)
    fluencyMatrix = getFluency(y)
    
    # Split data and train with firs 35%
    split = round(fluencyMatrix.size * 0.35)
    split2 = round(fluencyMatrix.size * 0.70)
    
    # List of feature indexes:
    featureIndexes = []
    for i in range(featureMatrix.shape[1]):
        featureIndexes.append(i)
    
    # Train model with all possible feature combinations, test the model,
    # calculate MAE and RMSE and save results
    combinations =  []
    for i in range(featureMatrix.shape[1]):
        # Create list of all combinations
        comb = list(itertools.combinations(featureIndexes, i+1))
        # Iterate through the list, train a model with selected features
        for j in range(len(comb)):
            model = fitLinearModel(featureMatrix[0:split, comb[j]], fluencyMatrix[0:split])
            predictions = model.predict(featureMatrix[split:split2, comb[j]])
            mu = predictions.mean()
            sigma = predictions.std()
            mae = np.sum(np.abs(predictions - fluencyMatrix[split:split2])) / predictions.size
            rmse = np.sqrt(np.sum(np.square(predictions - fluencyMatrix[split:split2])) / predictions.size)
            score = r2_score(predictions, fluencyMatrix[split:split2])
            result = [str(comb[j]), mae, rmse, score, mu, sigma]
            combinations.append(result)
            
    try:
        with open("features{}.csv".format(int(time.time())), 'w', newline='\n') as csvfile:
            wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            for i in combinations:
                wr.writerow(i)
        
        csvfile.close()        
        return combinations
    except FileExistsError:
        return combinations


# Test all feature combinations with random split and normal train set
def featureTestRandom(textgrid, pitch, y):
    featureMatrix = getFeatureMatrix(textgrid, pitch)
    fluencyMatrix = getFluency(y)
    
    # List of feature indexes:
    featureIndexes = []
    for i in range(featureMatrix.shape[1]):
        featureIndexes.append(i)

    X_train, X_test, y_train, y_test = train_test_split(featureMatrix, fluencyMatrix, test_size=0.3)
    
    # Train model with all possible feature combinations, test the model,
    # calculate MAE and RMSE and save results
    combinations =  []
    for i in range(featureMatrix.shape[1]):
        # Create list of all combinations
        comb = list(itertools.combinations(featureIndexes, i+1))
        # Iterate through the list, train a model with selected features
        for j in range(len(comb)):
            print(comb[j])
            model = LinearRegression.LinearRegression().fit(X_train[:, comb[j]], y_train)
            predictions = model.predict(X_test[:, comb[j]])
            mu = predictions.mean()
            mu_f = y_test.mean()
            sigma = predictions.std()
            sigma_f = y_test.std()
            mae = np.sum(np.abs(predictions - y_test) / predictions.size)
            rmse = np.sqrt(np.sum(np.square(predictions - y_test)) / predictions.size)
            score = r2_score(y_test, predictions)
            
            result = [str(comb[j]), mae, rmse, score, mu, mu_f, sigma, sigma_f]
            combinations.append(result)
            
    try:
        with open("features{}.csv".format(int(time.time())), 'w', newline='\n') as csvfile:
            wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            for i in combinations:
                wr.writerow(i)
        
        csvfile.close()        
        return combinations
    except:
        return combinations


def linearTestRandom(featureMatrix, X_train, X_test, y_train, y_test):    
    # List of feature indexes:
    featureIndexes = []
    for i in range(featureMatrix.shape[1]):
        featureIndexes.append(i)
    
    # Train model with all possible feature combinations, test the model,
    # calculate MAE and RMSE and save results
    combinations =  []
    for i in range(featureMatrix.shape[1]):
        # Create list of all combinations
        comb = list(itertools.combinations(featureIndexes, i+1))
        # Iterate through the list, train a model with selected features
        for j in range(len(comb)):
            print(comb[j])
            model = LinearRegression.LinearRegression().fit(X_train[:, comb[j]], y_train)
            predictions = model.predict(X_test[:, comb[j]])
            mu = predictions.mean()
            mu_f = y_test.mean()
            sigma = predictions.std()
            sigma_f = y_test.std()
            mae = np.sum(np.abs(predictions - y_test) / predictions.size)
            rmse = np.sqrt(np.sum(np.square(predictions - y_test)) / predictions.size)
            score = r2_score(y_test, predictions)
            
            result = [str(comb[j]), mae, rmse, score, mu, mu_f, sigma, sigma_f]
            combinations.append(result)
            
    try:
        with open("featureTest0212.csv", 'w', newline='\n') as csvfile:
            wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            for i in combinations:
                wr.writerow(i)
        
        csvfile.close()        
        return combinations
    except FileExistsError:
        return combinations


def logisticTestRandom(featureMatrix, X_train, X_test, y_train, y_test):
    # List of feature indexes:
    featureIndexes = []
    for i in range(featureMatrix.shape[1]):
        featureIndexes.append(i)
    
    # Train model with all possible feature combinations, test the model,
    # calculate MAE and RMSE and save results
    combinations =  []
    for i in range(featureMatrix.shape[1]):
        # Create list of all combinations
        comb = list(itertools.combinations(featureIndexes, i+1))
        # Iterate through the list, train a model with selected features
        for j in range(len(comb)):
            print(comb[j])
            model = LinearRegression.LogisticRegression(penalty='l2', class_weight='balaced', solver='lbfgs', multi_class='ovr').fit(X_train[:, comb[j]], y_train)
            
            predictions = model.predict(X_test[:, comb[j]])
            mu = predictions.mean()
            mu_f = y_test.mean()
            sigma = predictions.std()
            sigma_f = y_test.std()
            mae = np.sum(np.abs(predictions - y_test) / predictions.size)
            rmse = np.sqrt(np.sum(np.square(predictions - y_test)) / predictions.size)
            score = r2_score(y_test, predictions)
            
            result = [str(comb[j]), mae, rmse, score, mu, mu_f, sigma, sigma_f]
            combinations.append(result)
            
    try:
        with open("logisticTest0212.csv", 'w', newline='\n') as csvfile:  
            wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            for i in combinations:
                wr.writerow(i)
        
        csvfile.close()        
        return combinations
    except FileExistsError:
        return combinations


def knnTestRandom(featureMatrix, X_train, X_test, y_train, y_test):
    # List of feature indexes:
    featureIndexes = []
    for i in range(featureMatrix.shape[1]):
        featureIndexes.append(i)
    
    # Train model with all possible feature combinations, test the model,
    # calculate MAE and RMSE and save results
    combinations =  []
    for i in range(featureMatrix.shape[1]):
        # Create list of all combinations
        comb = list(itertools.combinations(featureIndexes, i+1))
        # Iterate through the list, train a model with selected features
        for j in range(len(comb)):
            print(comb[j])
            model = KNeighborsClassifier(5, weights='distance', algorithm='auto').fit(X_train[:, comb[j]], y_train)

            predictions = model.predict(X_test[:, comb[j]])
            mu = predictions.mean()
            mu_f = y_test.mean()
            sigma = predictions.std()
            sigma_f = y_test.std()
            mae = np.sum(np.abs(predictions - y_test) / predictions.size)
            rmse = np.sqrt(np.sum(np.square(predictions - y_test)) / predictions.size)
            score = r2_score(y_test, predictions)
            
            result = [str(comb[j]), mae, rmse, score, mu, mu_f, sigma, sigma_f]
            combinations.append(result)
            
    try:
        with open("knnTest0212.csv", 'w', newline='\n') as csvfile:  
            wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            for i in combinations:
                wr.writerow(i)
        
        csvfile.close()        
        return combinations
    except FileExistsError:
        return combinations


def decisionTestRandom(featureMatrix, X_train, X_test, y_train, y_test):
    # List of feature indexes:
    featureIndexes = []
    for i in range(featureMatrix.shape[1]):
        featureIndexes.append(i)

    # Train model with all possible feature combinations, test the model,
    # calculate MAE and RMSE and save results
    combinations =  []
    for i in range(featureMatrix.shape[1]):
        # Create list of all combinations
        comb = list(itertools.combinations(featureIndexes, i+1))
        # Iterate through the list, train a model with selected features
        for j in range(len(comb)):
            print(comb[j])
            model = DecisionTreeClassifier().fit(X_train[:, comb[j]], y_train)
            
            predictions = model.predict(X_test[:, comb[j]])
            mu = predictions.mean()
            mu_f = y_test.mean()
            sigma = predictions.std()
            sigma_f = y_test.std()
            mae = np.sum(np.abs(predictions - y_test) / predictions.size)
            rmse = np.sqrt(np.sum(np.square(predictions - y_test)) / predictions.size)
            score = r2_score(y_test, predictions)
            
            result = [str(comb[j]), mae, rmse, score, mu, mu_f, sigma, sigma_f]
            combinations.append(result)
            
    try:
        with open("decisionTest0212.csv", 'w', newline='\n') as csvfile:
            wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            for i in combinations:
                wr.writerow(i)
        
        csvfile.close()        
        return combinations
    except FileExistsError:
        return combinations


def randomforestTestRandom(featureMatrix, X_train, X_test, y_train, y_test):
    # List of feature indexes:
    featureIndexes = []
    for i in range(featureMatrix.shape[1]):
        featureIndexes.append(i)

    # Train model with all possible feature combinations, test the model,
    # calculate MAE and RMSE and save results
    combinations =  []
    for i in range(featureMatrix.shape[1]):
        # Create list of all combinations
        comb = list(itertools.combinations(featureIndexes, i+1))
        # Iterate through the list, train a model with selected features
        for j in range(len(comb)):
            print(comb[j])
            model = RandomForestClassifier(n_estimators=100).fit(X_train[:, comb[j]], y_train)
            
            predictions = model.predict(X_test[:, comb[j]])
            mu = predictions.mean()
            mu_f = y_test.mean()
            sigma = predictions.std()
            sigma_f = y_test.std()
            mae = np.sum(np.abs(predictions - y_test) / predictions.size)
            rmse = np.sqrt(np.sum(np.square(predictions - y_test)) / predictions.size)
            score = r2_score(y_test, predictions)
            
            result = [str(comb[j]), mae, rmse, score, mu, mu_f, sigma, sigma_f]
            combinations.append(result)
            
    try:
        with open("randomforestTest0212.csv", 'w', newline='\n') as csvfile:
            wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            for i in combinations:
                wr.writerow(i)
        
        csvfile.close()        
        return combinations
    except FileExistsError:
        return combinations


def MLPTestRandom(featureMatrix, X_train, X_test, y_train, y_test):    
    # List of feature indexes:
    featureIndexes = []
    for i in range(featureMatrix.shape[1]):
        featureIndexes.append(i)

    # Train model with all possible feature combinations, test the model,
    # calculate MAE and RMSE and save results
    combinations =  []
    for i in range(featureMatrix.shape[1]):
        # Create list of all combinations
        comb = list(itertools.combinations(featureIndexes, i+1))
        # Iterate through the list, train a model with selected features
        for j in range(len(comb)):
            print(comb[j])
            model = MLPClassifier(hidden_layer_sizes=(16, 100, 17), max_iter=100).fit(X_train[:, comb[j]], y_train)
            
            predictions = model.predict(X_test[:, comb[j]])
            mu = predictions.mean()
            mu_f = y_test.mean()
            sigma = predictions.std()
            sigma_f = y_test.std()
            mae = np.sum(np.abs(predictions - y_test) / predictions.size)
            rmse = np.sqrt(np.sum(np.square(predictions - y_test)) / predictions.size)
            score = r2_score(y_test, predictions)
            
            result = [str(comb[j]), mae, rmse, score, mu, mu_f, sigma, sigma_f]
            combinations.append(result)
            
    try:
        with open("MPLTest0212.csv", 'w', newline='\n') as csvfile: 
            wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            for i in combinations:
                wr.writerow(i)
        
        csvfile.close()        
        return combinations
    except FileExistsError:
        return combinations


def adaTestRandom(featureMatrix, X_train, X_test, y_train, y_test):
    # List of feature indexes:
    featureIndexes = []
    for i in range(featureMatrix.shape[1]):
        featureIndexes.append(i)

    # Train model with all possible feature combinations, test the model,
    # calculate MAE and RMSE and save results
    combinations =  []
    for i in range(featureMatrix.shape[1]):
        # Create list of all combinations
        comb = list(itertools.combinations(featureIndexes, i+1))
        # Iterate through the list, train a model with selected features
        for j in range(len(comb)):
            print(comb[j])
            model = AdaBoostClassifier(n_estimators=100).fit(X_train[:, comb[j]], y_train)
            
            predictions = model.predict(X_test[:, comb[j]])
            mu = predictions.mean()
            mu_f = y_test.mean()
            sigma = predictions.std()
            sigma_f = y_test.std()
            mae = np.sum(np.abs(predictions - y_test) / predictions.size)
            rmse = np.sqrt(np.sum(np.square(predictions - y_test)) / predictions.size)
            score = r2_score(y_test, predictions)
            
            result = [str(comb[j]), mae, rmse, score, mu, mu_f, sigma, sigma_f]
            combinations.append(result)
            
    try:
        with open("AdaTest0212.csv", 'w', newline='\n') as csvfile: 
            wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            for i in combinations:
                wr.writerow(i)
        
        csvfile.close()        
        return combinations
    except FileExistsError:
        return combinations


# Test all classifiers with all feature combinations. Split is done randomly
def classifierTest(textgrid, pitch, y):
    featureMatrix = getFeatureMatrix(textgrid, pitch)
    fluencyMatrix = getFluency(y)
    
    X_train, X_test, y_train, y_test = train_test_split(featureMatrix, fluencyMatrix, test_size=0.3)
    
    print("linear")
    linearTestRandom(featureMatrix, X_train, X_test, y_train, y_test)
    print("logistic")
    logisticTestRandom(featureMatrix, X_train, X_test, y_train, y_test)
    print("knn")
    knnTestRandom(featureMatrix, X_train, X_test, y_train, y_test)
    print("decision")
    decisionTestRandom(featureMatrix, X_train, X_test, y_train, y_test)
    print("random")
    randomforestTestRandom(featureMatrix, X_train, X_test, y_train, y_test)
    print("MLP")
    MLPTestRandom(featureMatrix, X_train, X_test, y_train, y_test)
    print("ADA")
    adaTestRandom(featureMatrix, X_train, X_test, y_train, y_test)


# Calculates features and plots every feature with respect to combined fluency assessments.
#   Arguments are paths to folders which contain respected files.
#   Argument y is path to file which contains fluency approximations.
def plotFeatures(textgrid, pitch, y):
    featureMatrix = getFeatureMatrix(textgrid, pitch)
    fluencyMatrix = getFluency(y)    
    
    for i in range(featureMatrix.shape[1]):
        plt.figure()
        x = featureMatrix[:,i]
        y = fluencyMatrix
        plt.scatter(x, y, marker=".")
        plt.title = str(i)
        plt.show()


# Calculates features and plots every feature with respect to all fluency assessments
def plotAllFeatures(textgrid, pitch, y):
    featureMatrix = getFeatureMatrix(textgrid, pitch)
    fluencyMatrix = getMultipleF(y)    
    
    for i in range(featureMatrix.shape[1]):
        x = featureMatrix[:,i]
        y = fluencyMatrix
        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(9, 9)
        fig.title = str(i)
        axs[0, 0].set_title('Painotus')
        axs[0, 0].scatter(x, y[:, 0], marker=".")
        axs[0, 1].set_title('Sujuvuus')
        axs[0, 1].scatter(x, y[:, 1], marker=".")
        axs[1, 0].set_title('Tahti')
        axs[1, 0].scatter(x, y[:, 2], marker=".")
        axs[1, 1].set_title('Tunneilmaisu')
        axs[1, 1].scatter(x, y[:, 3], marker=".")


# Try to guess fluency ratings
#   Uses normal distribute with mean and deviation calculated from real fluency
#   ratings. Prints out scoring.
def testGuessing(y):
    fluencyMatrix = getFluency(y)
    
    maes = []
    rmses = []
    
    for i in range(100):
        y_train, y_test = train_test_split(fluencyMatrix, test_size=0.3)
    
        predictions = np.random.normal(y_train.mean(), y_train.std(), y_test.size)
    
        mae = np.sum(np.abs(predictions - y_test)) / predictions.size
        maes.append(mae)
        print("Mean absolute error is ", mae, " for test data")
    
        rmse = np.sqrt(np.sum(np.square(predictions - y_test)) / predictions.size)
        rmses.append(rmse)
        print("Root-mean-square error is ", rmse, " for test data")
    
    
    mae = np.array(maes).mean()
    rmse = np.array(rmses).mean()
    
    return mae, rmse


# Try to guess fluency rating by using only the mean
def meanTest(y):
    fluencyMatrix = getFluency(y)
    
    maes = []
    rmses = []
    
    for i in range(100):
        y_train, y_test = train_test_split(fluencyMatrix, test_size=0.3)
    
        predictions = np.random.normal(y_train.mean(), 0, y_test.size)
    
        mae = np.sum(np.abs(predictions - y_test)) / predictions.size
        maes.append(mae)
        print("Mean absolute error is ", mae, " for test data")
    
        rmse = np.sqrt(np.sum(np.square(predictions - y_test)) / predictions.size)
        rmses.append(rmse)
        print("Root-mean-square error is ", rmse, " for test data")
    
    
    mae = np.array(maes).mean()
    rmse = np.array(rmses).mean()
    
    return mae, rmse



if __name__ == "__main__":	
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group()
    
    # Positional arguments
    group.add_argument('textgrid', help='TextGrid file that contains timestamps')
    group.add_argument('pitch', help='f0 file that contains pitch of speech')
    group.add_argument('y', help='csv file that contains fluency information')
    
    # Optional argument flags
    group.add_argument('-p', '--plot', help='Plot features in respect to combined ratings')
    group.add_argument('-pa', '--plotall', help='Plot features in respect to all ratings')
    group.add_argument('-m', '--test', help='Test all features for combined rating')
    group.add_argument('-m', '--testall', help='Test all features for all rating criteria')    
    
    ns = parser.parse_args()
    
    if ns.textgrid is None:
        print("Textgrid file missing!")
    elif ns.pitch is None:
        print("F0 file missing!")
    elif ns.y is None:
        print("Fluency rating file missing!")
        
    if ns.plot == True:
        plotFeatures(ns.textgrid, ns.pitch, ns.y)
    elif ns.plotall == True:
        plotAllFeatures(ns.textgrid, ns.pitch, ns.y)
    elif ns.test == True:
        testRandom(ns.textgrid, ns.pitch, ns.y)
    elif ns.testall == True:
        testFeaturesMultiple(ns.textgrid, ns.pitch, ns.y)
    else:
        models = trainMultiple(ns.textgrid, ns.pitch, ns.y)
        for fluencymodel in range(len(models)):
            with open("{}_{}".format(fluencymodel, time.time()), 'wb') as modelfile:
                pickle.dump(models[fluencymodel], modelfile)

