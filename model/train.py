import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import csv, sys
import matplotlib.pyplot as plt
from tensorboard_logger import configure, log_value, log_histogram, log_images,  Logger
from PIL import Image
import cv2

dates = []
prices = []
name = str(sys.argv[1]) if len(sys.argv) > 1 else 'SVM for stock Preiction'
kernel = str(sys.argv[2]) if len(sys.argv) > 2 else 'linear'
C = float(sys.argv[3]) if len(sys.argv) > 3 else 1e3
gamma = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
degree= int(sys.argv[5]) if len(sys.argv) > 5 else 2

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return


configure("runs/run-3", flush_secs=5)


if __name__ == "__main__":

    get_data('goog.csv')

    dates = np.reshape(dates,(len(dates), 1))

    svm = SVR(kernel= kernel, C= C, degree= degree, gamma=gamma)

    svm.fit(dates, prices)

    predictions = svm.predict(dates)

    (rmse, mae, r2) = eval_metrics(prices, predictions)
    plt.scatter(dates, prices, color= 'black', label= 'Data')
    plt.plot(dates,predictions, color= 'red', label= 'SVM model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('SVM with '+kernel+ ' kernel')
    plt.legend()
    plt.savefig('svm.png')

    log_value('RMSE', rmse)
    log_value('MAE', mae)
    log_value('R2', r2)

    img = cv2.imread('svm.png')

    log_images('Model-3',[img])