import pygame,sys

pygame.init()  #初始化init() 及设置
size = width,height = 800,600

speed = [1,1]
BLACK = 0,0,0

screen = pygame.display.set_mode(size)

icon = pygame.image.load("image/dog01.jpg")#加载图片
pygame.display.set_icon(icon) #图标的使用

pygame.display.set_caption("example")  #游戏开始的首标题设置
ball = pygame.image.load("image/cat.jpg") #
ballrect = ball.get_rect()
fps = 300
fclock = pygame.time.Clock()
# def load_module():
    

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif  event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                speed[0] = speed[0] if speed[0] == 0 else (abs(speed[0]) - 1)*int(speed[0]/abs(speed[0]))
            elif event.key == pygame.K_RIGHT:
                speed[0] = speed[0] +1 if speed[0] >0 else speed[0]-1
            elif event.key == pygame.K_UP:
                speed[1] = speed[1] +1 if speed[1] >0 else speed[1]-1
            elif event.key == pygame.K_DOWN:
                speed[1] = speed[1] if speed[1] == 0 else (abs(speed[1]) - 1)*int(speed[1]/abs(speed[1]))
            elif event.key == pygame.ESCAPE:
                sys.exit()