import numpy as np
from numpy import pi
def wrap_pi(angle):
    a = (angle + np.pi) % (2 * np.pi)

    return a - np.pi


# def angle_wrt_home(home, body):

#     p = body.position - Vec2d(home)
#     v = body.velocity
#     phi = pi - (body.angle - p.angle)
#     d_phi = -(body.angular_velocity - (p.x * v.y - p.y * v.x) / (p.length ** 2))
#     ret = (phi, d_phi)
#     return ret 


def wrap_2pi(angle):
    a = angle % 2 * np.pi
    if a < 0:
        a += 2 * np.pi
    return a

def angle_wrt_home(body,angle,home):
    """
    Gets the angle of the body w.r.t the another body, home
    parameters:
    ----------
    body: float [x,y]
        position of the body
    angle: float
        angle of the body
    home: float [x,y]
        position of the 
    
    returns:
    --------
    phi: float
        the angle
    """


    p = body -home
    phi = pi - (angle - np.arctan2(p[1], p[0]))
    return phi