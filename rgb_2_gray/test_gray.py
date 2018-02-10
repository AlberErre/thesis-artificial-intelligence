
imagen = open('test.jpg')
gray = Color2Gray(imagen)

def Color2Gray(color_image): # Como en computer vision, usamos RGB para cambiar la imagen a tono de grises y poder procesarla
    imagen_rgb = scipy.misc.imresize(color_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interp='bilinear')
    red = imagen_rgb[:,:,0] # 3 elemento es el filtro de color RGB
    green = imagen_rgb[:,:,1]
    blue = imagen_rgb[:,:,2]
    gray_color = ((0.2989*red) + (0.5870*green) + (0.1140*blue))/128
    gray_image = gray_color.astype('float32')/(128-1)

    plt.imshow(gray_image) 
    plt.show()

    """
    rgb2k = np.array([0.299, 0.587, 0.114])
    # gray_image = np.round(np.sum(color_image * rgb2k, axis=-1)).astype('uint8')
    gray_image = np.sum(color_image * rgb2k, axis=-1)/255 # Otra forma, esta es float

    """
    return gray_image 

