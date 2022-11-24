import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, os.path
from matplotlib import image
import matplotlib.pyplot as plt
from numpy import asarray

#from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
#from keras.models import Sequential, model_from_json
#from keras.utils import to_categorical
from os.path import isfile, join
#from keras import backend as K
from os import listdir
from PIL import Image
num_img_from_each_index=300
index_by_directory = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'add': 10,
    'sub': 11,
    'exp': 12,
    'div': 13
}
def get_index_by_directory(directory):
   return index_by_directory[directory]
def imgpreprocess(img):
    img = ~img
    if img is not None:
     _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  #thresh binary- pixel below 127 be 0 and above be 1
     ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
     cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])#bounding rect makes a rec around digit
     r = 0                                                   #returs x,y,w,h in order. sort func sorts by
     for i in cnt:                                           #x in ascending order and retunrs the cnt,
         x, y, w, h = cv2.boundingRect(i)                    # key lambda is used to give a func in sort()
         r = max(w * h, r)#selects conotur with most area    #this shit is done to remove possible noise
         if r == w * h:                         #(dataset is clean and tidy) but in test, we will pass sliced imgs
             x_max = x                          #to this func for preprocessing which may have noise (like dots)
             y_max = y
             w_max = w
             h_max = h
     im_crop = thresh[y_max:y_max + h_max + 10, x_max:x_max + w_max + 10]  # Crop the image as most as possible
     im_resize = cv2.resize(im_crop, (28, 28))  # Resize to (28, 28)
     #im_resize = np.reshape(im_resize, (784, 1))
     return im_resize
def load_imgs(folder):
    train_data=[]
    i=0
    for filename in os.listdir(folder):
        #print("working now")
        if i==num_img_from_each_index:
          print("breaking now")
          break
        i=i+1
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
         train_data.append(imgpreprocess(img))
    print("worked")
    return train_data
#folder=datasets
def load_all_imgs():
    #dataset_dir = "./datasets/"
    train_folder = "/Users/dhruva/PycharmProjects/solvertry2/datasets/train/"
    data_list = listdir(train_folder)
    initial_data = True
    data = []

    print('Exporting images...')
    for index in data_list:
        if index.startswith('.'):
            continue
        print(index)
        if initial_data:
            initial_data = False
            data = load_imgs(train_folder + index)
            for i in range(0, len(data)):
                data[i] = np.append(data[i], [str(get_index_by_directory(index))])
            continue

        aux_data = load_imgs(train_folder + index)
        for i in range(0, len(aux_data)):
            aux_data[i] = np.append(aux_data[i], [str(get_index_by_directory(index))])
        data = np.concatenate((data, aux_data))
    print(data.shape)
    df=pd.DataFrame(data,index=None)
    df.to_csv('train_data.csv',index=False)


######################
def extract_imgs(img):
    img = ~img  # Invert the bits of image 255 -> 0
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # Set bits > 127 to 1 and <= 127 to 0
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])  # Sort by x

    img_data = []
    rects = []
    for c in cnt:
        x, y, w, h = cv2.boundingRect(c)
        rect = [x, y, w, h]
        rects.append(rect)

    bool_rect = []
    # Check when two rectangles collide
    for r in rects:
        l = []
        for rec in rects:
            flag = 0
            if rec != r:
                if r[0] < (rec[0] + rec[2] + 10) and rec[0] < (r[0] + r[2] + 10) and r[1] < (rec[1] + rec[3] + 10) and \
                        rec[1] < (r[1] + r[3] + 10):
                    flag = 1
                l.append(flag)
            else:
                l.append(0)
        bool_rect.append(l)

    dump_rect = []
    # Discard the small collide rectangle
    for i in range(0, len(cnt)):
        for j in range(0, len(cnt)):
            if bool_rect[i][j] == 1:
                area1 = rects[i][2] * rects[i][3]
                area2 = rects[j][2] * rects[j][3]
                if (area1 == min(area1, area2)):
                    dump_rect.append(rects[i])

    # Get the final rectangles
    final_rect = [i for i in rects if i not in dump_rect]
    for r in final_rect:
        x = r[0]
        y = r[1]
        w = r[2]
        h = r[3]

        im_crop = thresh[y:y + h + 10, x:x + w + 10]  # Crop the image as most as possible
        im_resize = cv2.resize(im_crop, (28, 28))  # Resize to (28, 28)
        # im_resize = np.reshape(im_resize, (1,28, 28)) # Flat the matrix
        img_data.append(im_resize)
        # cv2.imshow("__", im_resize)
        # plt.imshow(im_resize, cmap='gray')
        # plt.show()
        yield im_resize
        # plt.imshow(im_resize, cmap='gray')
        # plt.show()


###################
if not os.path.exists('train_data.csv'):
  load_all_imgs()
csv_train_data = pd.read_csv('train_data.csv', index_col=False)
y_train = csv_train_data[['784']]
csv_train_data.drop(csv_train_data.columns[[784]], axis=1, inplace=True)
csv_train_data.head()
y_train = np.array(y_train)
x_train = []
for i in range(len(csv_train_data)):
    o=np.array(csv_train_data[i:i+1]).reshape(1, 28, 28)
    x_train=np.append(x_train,o)
    #print("working")
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (-1, 28, 28, 1))

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2])
y_train=y_train.reshape(y_train.shape[0])
print(x_train.shape)
print(y_train.shape)
print('Training model...')

#############################
class Convulation:
    def __init__(self, nf, n):
        self.nf = nf
        self.n = n
        self.filter = np.random.randn(nf, n, n) / (n * n)  # will initial a normalized 3D matrix with random numbers

    def patch(self, img):
        h, w = img.shape
        self.img = img
        for i in range(h - self.n + 1):
            for j in range(w - self.n + 1):
                img_patch = img[i:i + self.n, j:j + self.n]
                yield img_patch, i, j

    def fwdprop(self, img):
        h, w = img.shape
        Convulation_output = np.zeros((h - self.n + 1, w - self.n + 1, self.nf))
        for img_patch, i, j in self.patch(img):
            Convulation_output[i, j] = np.sum(img_patch * self.filter, axis=(1, 2))
        return Convulation_output

    def backprop(self, dL_dout, lrate):
        dl_dF_params = np.zeros(self.filter.shape)
        for img_patch, i, j in self.patch(self.img):
            for k in range(self.n):
                dl_dF_params[k] += img_patch * dL_dout[i, j, k]
        self.filter -= lrate * dl_dF_params
        return dl_dF_params


"""      
conn = Convulation(18, 7)
out = conn.fwdprop(img)
print(out.shape)
plt.imshow(out[:, :, 17], cmap='gray')
plt.show()
"""


class Max_Pool:
    def __init__(self, n):
        self.n = n

    def patch(self, img):
        h2 = img.shape[0] // self.n
        w2 = img.shape[1] // self.n
        self.img = img

        for i in range(h2):
            for j in range(w2):
                img_patch = img[(i * self.n): (i * self.n + self.n), (j * self.n): (j * self.n + self.n), ]
                yield img_patch, i, j

    def fwd_prop(self, img):
        h, w, nf = img.shape
        output = np.zeros((h // self.n, w // self.n, nf))
        for img_patch, i, j in self.patch(img):
            output[i, j] = np.amax(img_patch, axis=(0, 1))
        return output

    def back_prop(self, dL_dout):
        dL_dmax_pool = np.zeros(self.img.shape)
        for img_patch, i, j, in self.patch(self.img):
            h, w, nf = img_patch.shape
            maximum_val = np.amax(img_patch, axis=(0, 1))

            for i1 in range(h):
                for j1 in range(w):
                    for k1 in range(nf):
                        if img_patch[i1, j1, k1] == maximum_val[k1]:
                            dL_dmax_pool[i * self.n + i1, j * self.n + j1, k1] = dL_dout[i, j, k1]
        return dL_dmax_pool


"""
conn2= Max_Pool(4)
out2= conn2.fwd_prop(out)
print(out2.shape)
plt.imshow(out2[:, :, 17], cmap='gray')
plt.show()  
"""


class Softmax:
    def __init__(self, input_node, softmax_node):
        self.weight = np.random.randn(input_node, softmax_node) / input_node
        self.bias = np.zeros(softmax_node)

    def forward_prop(self, image):
        self.orig_im_shape = image.shape
        image_modified = image.flatten()
        self.modified_input = image_modified
        output_val = np.dot(image_modified, self.weight) + self.bias
        self.out = output_val
        exp_out = np.exp(output_val)
        return exp_out / np.sum(exp_out, axis=0)

    def back_prop(self, dL_dout, learning_rate):
        for i, grad in enumerate(dL_dout):
            if grad == 0:
                continue

            transformation_eq = np.exp(self.out)
            S_total = np.sum(transformation_eq)

            dy_dz = -transformation_eq[i] * transformation_eq / (S_total ** 2)
            dy_dz[i] = transformation_eq[i] * (S_total - transformation_eq[i]) / (S_total ** 2)

            dz_dw = self.modified_input
            dz_db = 1
            dz_d_inp = self.weight
            dL_dz = grad * dy_dz
            dL_dw = dz_dw[np.newaxis].T @ dL_dz[np.newaxis]
            dL_db = dL_dz * dz_db
            dL_d_inp = dz_d_inp @ dL_dz
            self.weight -= learning_rate * dL_dw
            self.bias -= learning_rate * dL_db

            return dL_d_inp.reshape(self.orig_im_shape)


class Softmax:
    def __init__(self, input_node, softmax_node):
        self.weight = np.random.randn(input_node, softmax_node) / input_node
        self.bias = np.zeros(softmax_node)

    def fwd_prop(self, image):
        self.orig_im_shape = image.shape
        image_modified = image.flatten()
        self.modified_input = image_modified
        output_val = np.dot(image_modified, self.weight) + self.bias
        self.out = output_val
        exp_out = np.exp(output_val)
        return exp_out / np.sum(exp_out, axis=0)

    def back_prop(self, dL_dout, learning_rate):
        for i, grad in enumerate(dL_dout):
            if grad == 0:
                continue

            transformation_eq = np.exp(self.out)
            S_total = np.sum(transformation_eq)

            dy_dz = -transformation_eq[i] * transformation_eq / (S_total ** 2)
            dy_dz[i] = transformation_eq[i] * (S_total - transformation_eq[i]) / (S_total ** 2)

            dz_dw = self.modified_input
            dz_db = 1
            dz_d_inp = self.weight
            dL_dz = grad * dy_dz
            dL_dw = dz_dw[np.newaxis].T @ dL_dz[np.newaxis]
            dL_db = dL_dz * dz_db
            dL_d_inp = dz_d_inp @ dL_dz
            self.weight -= learning_rate * dL_dw
            self.bias -= learning_rate * dL_db

            return dL_d_inp.reshape(self.orig_im_shape)


#conn3 = Softmax(18 * 393 * 268, 10)
#out3 = conn3.fwd_prop(out2)
#print(out3)
#x = 6900
train_images = x_train
train_labels = y_train
#test_images = X_test[:x]
#test_labels = Y_test[:x]
conv = Convulation(8, 3)
pool = Max_Pool(2)
softmax = Softmax(13 * 13 * 8, 14)


def cnn_forward_prop(image, label):
    out_p = conv.fwdprop((image / 255) - 0.5)
    out_p = pool.fwd_prop(out_p)
    out_p = softmax.fwd_prop(out_p)
    cross_ent_loss = -np.log(out_p[label]) if out_p[label] > 0 else 0
    accuracy_eval = 1 if np.argmax(out_p) == label else 0
    return out_p, cross_ent_loss, accuracy_eval


def training_cnn(image, label, learn_rate=0.005):
    out, loss, acc = cnn_forward_prop(image, label)
    gradient = np.zeros(14)
    gradient[label] = -1 / out[label]

    grad_back = softmax.back_prop(gradient, learn_rate)
    grad_back = pool.back_prop(grad_back)
    grad_back = conv.backprop(grad_back, learn_rate)

    return loss, acc


for epoch1 in range(4):
    print('Epoch %d ---->' % (epoch1 + 1))

    shuffle_data = np.random.permutation(len(train_images))
    train_images = train_images[shuffle_data]
    train_labels = train_labels[shuffle_data]

    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i % 100 == 0:
            print('step')
            print((i + 1) / 100)
            print('avg loss: ')
            print(loss / 100)
            print('accuracy')
            print(num_correct)
            print('\n')
            loss = 0
            num_correct = 0
        l1, accu = training_cnn(im, label)
        loss += l1
        num_correct += accu
"""        
print('*TestingPhase*')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l1, accu = cnn_forward_prop(im, label)
    loss += l1
    num_correct += accu
num_tests = len(test_images)

print('Loss:', loss / num_tests)
print('Accuracy:', num_correct / num_tests)
"""
y = 69
plt.imshow(train_images[y], cmap='gray')
#plt.show()
cv2.imshow("__",train_images[y] )
print(train_images[y].shape)
out_p = conv.fwdprop((train_images[y] / 255) - 0.5)
out_p = pool.fwd_prop(out_p)
out_p = softmax.fwd_prop(out_p)
print("image is probably", np.argmax(out_p))
print("image is really", train_labels[y])
#def predict(img):

"""
img = cv2.imread('/content/sample_data/5.png', cv2.IMREAD_GRAYSCALE)/255
def predict(img1):
  img = cv2.resize(img1, (28, 28))
  plt.imshow(img, cmap='gray')
  plt.show()
  print(img.shape)
  out_p= conv.fwdprop((img/255)-0.5)
  out_p= pool.fwd_prop(out_p)
  out_p=softmax.fwd_prop(out_p)
  print("image is probably",np.argmax(out_p))
predict(img)
"""





eq=''
img1 = cv2.imread('/Users/dhruva/PycharmProjects/solvertry2/datasets/train/0/0_62375.jpg', cv2.IMREAD_GRAYSCALE)
img1 = cv2.resize(img1, (28, 28))
result=img1
img = cv2.imread('/Users/dhruva/PycharmProjects/solvertry2/eq1.png', cv2.IMREAD_GRAYSCALE)
if img is not None:
    #img_data = extract_imgs(img)
    #print(len(img_data))
    #print(img_data[0].shape)
    #digit=np.reshape(img_data[0],  (28, 28))
    #digit=np.array(img_data[0])
    #digit = img_data[0]
    #print(digit.shape)
    #digit=digit.reshape(28, 28)
    #digit=np.squeeze(digit, axis=0).shape
    #digit = np.array(digit)
    #digit = np.squeeze(digit, axis=(0,3)).shape
    #digit = np.array(digit)
    #print(digit.shape)
    #plt.imshow(digit, cmap='gray')
    #plt.show()
    #for i in range(len(img_data)):

    for img in extract_imgs(img):
        #img_data = extract_imgs(img)
        #img_data = np.array(img_data)
        print(img.shape)
        result = np.concatenate((result, img), axis=1)
        img=img/255
        img=img-0.5
        out_1 = conv.fwdprop(img)
        out_1 = pool.fwd_prop(out_1)
        out_1 = softmax.fwd_prop(out_1)
        x=np.argmax(out_1)
        print("image is probably", x)
        #return np.argmax(out_p)
        if x == 10:
            eq += '+'
        elif x == 11:
            eq += '-'
        elif x == 12:
            eq += 'x'
        elif x == 13:
            eq += '/'    
        else:
            eq += str(x)
        #img_data[i] = np.array(img_data[i])
        #img_data[i] = img_data[i].reshape(-1, 28, 28, 1)
        #plt.imshow(img, cmap='gray')
        #plt.show()

cv2.imshow("result", result)
print("equation is probably", eq)
img1 = cv2.imread('/Users/dhruva/PycharmProjects/solvertry2/datasets/train/0/0_62375.jpg', cv2.IMREAD_GRAYSCALE)
#cv2.waitKey(0)
img1 = cv2.resize(img1, (28, 28))
img2 = cv2.resize(img,(28,28))
img1 = np.concatenate((img1, img2), axis=1)

#img1 = np.reshape(im_resize, (1,28, 28))
#img2= np.reshape(im_resize, (1,28, 28))
#lgt=len(os.listdir('/Users/dhruva/PycharmProjects/solvertry2/datasets/train/add'))-1
#print(lgt)
#img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
cv2.imshow("__",img1)
plt.show()
cv2.waitKey(0)
#plt.show()
