# autocv
2nd Prize Code in AutoCV Challenge

You should refer this link(https://github.com/HeungChangLee/autodl_starting_kit_stable).
And you should download pretrained weights of MobileNet_v2(https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz).

Our algorithm has flexible architecture for the various size of various inputs. This architecture is based on MobileNet_v2 which achieved 71.9 top-1 accuracy on the ImageNet. And the weight initialization is always constant because of pre-trained weights, so the variance of reproduced results is smaller than random initialized methods.
Our main insight is that when you modify MobileNet_v2, you do not change stride. We experimented with changing the stride for a lighter model, but we discover from various experiments that the pre-trained weights on ImageNet are broken if the stride is changed.
