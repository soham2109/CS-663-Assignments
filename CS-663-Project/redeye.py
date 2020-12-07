def redeye(flash_image,noflash_image):
    

    flash_rgb = cv2.cvtColor(flash_image, cv2.COLOR_BGR2RGB).astype('double')/255
    noflash_rgb = cv2.cvtColor(noflash_image, cv2.COLOR_BGR2RGB).astype('double')/255

    [h,w,_] = flash_image.shape
    noflash_image = cv2.cvtColor(noflash_image, cv2.COLOR_BGR2YCR_CB)
    flash_image = cv2.cvtColor(flash_image, cv2.COLOR_BGR2YCR_CB)

    noflash_image = noflash_image.astype('double')
    flash_image = flash_image.astype('double')

    R = flash_image[:,:,1] - noflash_image[:,:,1]
    R = R - np.min(R)
    R = R/np.max(R)

    flag = np.zeros(R.shape, np.uint8)
    Rm = np.mean(R)
    Rdev = np.std(R)
    thr = max(0.6, Rm+(Rdev*3))

    flag[R>thr] = 1
    flag[(noflash_image[:,:,0]/255) >= thr] = 0


    maskff = np.zeros((flag.shape[0]+2, flag.shape[1]+2), np.uint8)
    cv2.floodFill(flag, maskff, (0,0), 1)
    maskff = 1 - maskff
    se = disk(2)
    maskff = cv2.erode(maskff, se,iterations=1)
    maskff = cv2.dilate(maskff, se, iterations=1)


    [y,x] = np.nonzero(maskff)
    im = np.copy(flash_rgb )

    bias=[26,49]
    for i in range(0, y.shape[0]):
        f = 0
        if y[i]<50 or x[i]<50 or x[i] > w-50 or y[i] > h-50:
            f = 1
        else:
            mask = maskff[y[i]-bias[1]:y[i]+bias[1],x[i]-bias[1]:x[i]+bias[1]]
            mask[23:-23,23:-23] = 0
            if np.any(mask == 1) == 1:
                f=1
        if f==0:
            im[y[i], x[i], 0] = 0.8*(0.299*flash_rgb [y[i], x[i], 0]+ 0.587*flash_rgb [y[i], x[i], 1] + 0.114*flash_rgb [y[i], x[i],2])

    return flash_rgb,noflash_rgb,im
    

flash_image = cv2.imread('flash.jpg')
noflash_image = cv2.imread('no_flash.jpg')

flash_rgb,noflash_rgb,image_noredeye = redeye(flash_image,noflash_image)

fig = plt.figure(figsize=[16, 16])
fig.add_subplot(1,2,1) 
plt.imshow(flash_rgb ) 
plt.title('input image')
fig.add_subplot(1,2,2)
plt.imshow(image_noredeye) 
plt.title('red eyes removed')

fig = plt.figure(figsize=[8, 8])
plt.imshow(noflash_rgb , cmap='gray')
plt.title('ambient image')
