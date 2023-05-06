from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

def EvalModel(y_pred,y_test):
    # Calculer la corrélation de Spearman entre les prédictions et les valeurs réelles
    spearman_corr, _ = spearmanr(y_pred, y_test)
    print('Spearman correlation: %.3f' % spearman_corr)

    # Calculer le coefficient de détermination R2
    r2 = r2_score(y_test, y_pred)
    print('Coefficient de determination R^2: %.3f' % r2)

    # Calculer l'erreur quadratique moyenne (RMSE)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print('Erreur quadratique moyenne (RMSE): %.3f' % rmse)