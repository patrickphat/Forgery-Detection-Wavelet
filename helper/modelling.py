def count_tampered(model,image):
  batches = img2batch(image)
  label = []
  for batch in batches:
    feature = np.array(extract_features(batch))
    feature = np.expand_dims(feature,0)
    feature = scaler_obj.transform(feature)
    y_hat = (model.predict(feature)[0][0]>0.5)*1
    label.append(y_hat)
  ones = collections.Counter(label)[1]
  return ones