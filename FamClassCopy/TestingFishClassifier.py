TestingFishClassifier
# Testing the neural network to do stuff



#  Load the model: give the full directory name using os
print('Loading Trained Model')
new_model = load_model(os.path.join('models','famClassModel.h5'))

print('Classifying new image')
yhatnew_test = new_model.predict(np.expand_dims(resize/255, 0))
# if yhatnew[10] > 0.8:
#     print(f'Predicted: Port Jackson')
# else:
#     print(f'Predicted: Not PJ')

# position 9 should be highest as this represents port jackson shark
print('****************************************************************************************************')
print('****************************************************************************************************')
print('---- ID Num', np.argmax(yhatnew_test)+1, ', Probability: ', yhatnew_test.max())
# print('---- Checksum for Probability: ', np.sum(yhatnew_test))
print('****************************************************************************************************')
print('****************************************************************************************************')

# plot show at the end! 
# Halts the program to ensure figures stay viewable with program paused and not ended
plt.show()