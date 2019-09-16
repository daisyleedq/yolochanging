Problems is why I am doing synchronous training. Before I tested, in asychronous training, the worker0 is faster than worker1.
        Strategies：
        1.Calculate single machine的fps
        2.Calculate single machine with input pipe line
        3.Calculate distributed training, two machines without inputpipeline
        4.Calculate distributed training with Inputpipeline。Training process itself is slow, now adding the time of transporting the model, inputpipeline leaded improvment is not so obvious for big models, but we could expect mobilenetssd could be good.
        5.Calculate inference with and without pipeline.  I personally think there is no meaning to do distributed inference.
        4.Will the model size influence the training speed? Yes, but it is hard to make comparisons we use different platforms. because tensorflow didnt provide group_convolution operator? paddlepaddle we may compare infrence with inputpipeline and without pipeline. But there are two different parallelism in training.

