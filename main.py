from random import random
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.core.window import Window
import numpy as np
from test import grad_descend,load_weights,forward_prop

def antiali(img,dimx=28,dimy=28):
    ksize = 3
    nimg = np.zeros(img.shape)
    for x in range(0,dimx):
        for y in range(0,dimy):
            val = 0
            num = 0
            for i in range(-ksize+1,ksize):
                if x+i < dimx and x+i > 0:
                    for j in range(-ksize+1,ksize):
                        if y+j < dimx and y+j > 0:
                            val = val + img[y+j,x+i]
                            num = num + 1
            val = val/num
            nimg[y,x] = (val + img[y,x])*2
    return nimg

class MyPaintWidget(Widget):
    digit = np.zeros((28,28))
    theta = []
    def redraw(self):
        (w,h) = Window.size
        self.canvas.clear()
        with self.canvas:
            for x in range(0,28):
                for y in range(0,28):
                    if self.digit[y,x]!=0:
                        color = (self.digit[y,x],0,0)
                        Color(*color, mode='rgb')
                        Rectangle(pos=((((x)/28))*w,((((28 - y)/28))*h)),size=(w//28,h//28))

    def on_touch_down(self, touch):
        tx = int((touch.x/Window.size[0])*28)
        ty = int(28-((touch.y/Window.size[1])*28))
        self.digit[ty,tx] = 1
        self.redraw()
        self.canvas.ask_update()

    def on_touch_move(self, touch):
        # touch.ud['line'].points += [touch.x, touch.y]
        tx = int((touch.x/Window.size[0])*28)
        ty = int(28-((touch.y/Window.size[1])*28))
        self.digit[ty,tx] = 1
        self.redraw()
        self.canvas.ask_update()
        

class MyPaintApp(App):
    def build(self):
        parent = Widget()
        self.painter = MyPaintWidget()
        clearbtn = Button(text='Clear')
        clearbtn.bind(on_release=self.clear_canvas)
        predbut = Button(text='Predict',pos=(100,0))
        predbut.bind(on_release=self.predict)
        self.l = Label(text="Prediction: ",pos=(200,0))
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(predbut)
        parent.add_widget(self.l)
        self.load()
        return parent

    def clear_canvas(self, obj):
        self.painter.canvas.clear()
        self.painter.digit = np.zeros((28,28))
        self.l.text = "Prediction: "
    def load(self):
        self.painter.theta = load_weights() 

    def predict(self,obj):
        
        self.painter.digit = antiali(self.painter.digit)
        self.painter.redraw()
        X = [np.reshape(self.painter.digit,28*28)]
        a = forward_prop(self.painter.theta,X)
        self.l.text="Prediction: " + str(np.argmax(np.asarray(a)[3][:,0]))
        

if __name__ == '__main__':
    MyPaintApp().run()