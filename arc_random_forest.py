from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

model = RandomForestRegressor(bootstrap=True, 
                               #max_features='log2', 
                               #min_samples_leaf=15, 
                               #min_samples_split=2,
                               #criterion='entropy',
                               #class_weight='balanced',
                               n_estimators=100, 
                               n_jobs=-1, 
                               random_state=42
                               )


score = []
for i in range(len(raw_train)):
    print(f'File : {i}')
    X_train, y_train = raw_train[i][0], raw_train[i][1]
    X_val, y_val = raw_eval[0][0], raw_eval[0][1]
    if X_train.shape == X_val.shape and y_train.shape == y_val.shape:
        model.fit(X_train.reshape(1, -1), y_train.reshape(1, -1))
        prediction = model.predict(X_val.reshape(1, -1))
        score_ = mean_squared_error(y_val.reshape(1, -1), prediction)
        print(score_)
        score.append(score_)
#models.append(model)
print(np.mean(score))

pred = model.predict(raw_eval[0][0].reshape(1, -1))
display(pred.reshape(3, 3))

for i in range(pred.shape[0]):
    pred[i] = np.around(pred[i])
    
pred = pred.reshape(3, 3)
display(pred)
display(raw_eval[0][1])

cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
fig, axs = plt.subplots(1, 3, figsize=(15,15))
axs[0].imshow(raw_eval[0][0], cmap=cmap, norm=norm)
axs[0].axis('off')
axs[0].set_title('Train Input')
axs[1].imshow(raw_eval[0][1], cmap=cmap, norm=norm)
axs[1].axis('off')
axs[1].set_title('Train Output')
axs[2].imshow(pred, cmap=cmap, norm=norm)
axs[2].axis('off')
axs[2].set_title('Train Predict')
plt.show();
