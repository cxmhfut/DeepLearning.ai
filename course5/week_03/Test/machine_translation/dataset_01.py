from course5.week_03.Test.machine_translation.nmt_utils import *

m = 10000
dataset,human_vocab,machine_vocab,inv_vocab = load_dataset(m)

#print(dataset[:10])

Tx = 30
Ty = 10
X,Y,Xoh,Yoh = preprocess_data(dataset,human_vocab,machine_vocab,Tx,Ty)

print(X.shape)
print(Y.shape)
print(Xoh.shape)
print(Yoh.shape)

index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])