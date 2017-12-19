import numpy as np
import cv2
import rpy2.robjects as robjects

# Set to primary Mac Webcam
cap = cv2.VideoCapture(0)

def hue_extract(hsv_matrix):
    hue_only_matrix = []
    for row in hsv_matrix:
        hue_only_row = []
        for pixel in row:
            hue_only_row.append(int(pixel[0]))
        hue_only_matrix.append(hue_only_row)
    return hue_only_matrix

robjects.r['load']('weights.rda')

robjects.r('''
    # Activation Function
    sigmoid <- function(z){
      return(1/(1 + exp(-z)))
    }

    # Feed Forward Function
    forward <- function(X, Weights, length_l1, p, k){
      # Weights
      Theta1 <- matrix(Weights[1:(length_l1*(p+1))],nrow=length_l1)
      Theta2 <- matrix(Weights[((length_l1*(p+1))+1):length(Weights)],nrow=k)

      # Including intercept in X and transposing the matrix
      n <- nrow(X)
      if(is.null(n)) n<- 1
      X_NN <- c(rep(1,n),unlist(X))

      # Computaring neurons values and output
      l1 <- sigmoid(X_NN%*%t(Theta1))
   
      l1_i <- cbind(rep(1,n),l1)
      l2 <- sigmoid(l1_i%*%t(Theta2))

      M <- apply(l2,1,max)
      Y_classified <- floor(l2/M)

      return(list( Theta1=Theta1, Theta2=Theta2, l1=l1, l2= l2[1,], Y_classified=Y_classified[1,]))
    }

''')

weights = robjects.r['Weights_backp']
forward = robjects.r['forward']


while True:
    # Read in Frame

    ret, frame = cap.read()

    if cv2.waitKey(1) == 99: #99 is the ord key of 'c'
        # Selection coordinates
        r = cv2.selectROI(frame)

        # Cropping image
        cropped_frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        # Conversion to hsv
        resized_img = cv2.resize(cropped_frame,(10,10))
        hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
        hue_matrix = hue_extract(hsv_img)

        # Flatten Matrix and normalization
        inp=[]
        for row in hue_matrix:
            for value in row:
                inp.append(value/179) #0-179 is the hue range
        # Feed Forward
        result = forward(inp, weights, 35, 100, 3)
        classified = [int(x) for x in result[4]]

        # De-coding result
        encodings = {'Blue': [0,0,1], 'Red': [0,1,0], 'Green': [1,0,0]}
        for color in encodings:
            if encodings[color] == classified:
                answer = color
                break

        # Print Results
        cv2.putText(cropped_frame, answer, (0,50), cv2.FONT_HERSHEY_PLAIN, 3, 255, 3)
        cv2.imshow("crop", cropped_frame)

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
