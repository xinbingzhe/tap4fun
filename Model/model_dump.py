from sklearn.externals import joblib

def model_dump(clf,model_pth):
    joblib.dump(clf,model_pth)
    print(' model dump complete ')

def model_load(model_pth):
    return joblib.load(model_pth)