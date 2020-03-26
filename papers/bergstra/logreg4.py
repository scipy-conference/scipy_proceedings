N = 4
feats = 100
D = (numpy.random.randn(N, feats),
     numpy.random.randint(size=N,low=0, high=2))
training_steps = 10
for i in range(training_steps):
    pred, err = train(D[0], D[1])
print "Final model:",
print w.get_value(), b.get_value()
print "target values for D", D[1]
print "prediction on D", predict(D[0])
