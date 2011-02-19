predict = function(inputs=[x], 
                   outputs=prediction)
train = function(
            inputs=[x,y],
            outputs=[prediction, xent],
            updates={w:w-0.1*gw, b:b-0.1*gb})
