glcompat_attrs=14 //FLAG_VERTCOLORS+FLAG_TEXTURE+FLAG_FOG

[VERTEX]

#include "common.glsl"

IN mediump vec4 a_vert;
IN vec4 a_color;
OUT mediump vec4 v_color;
IN vec4 a_texcoord;
OUT mediump vec2 v_texcoord;

void main()
{
gl_Position = u_projmodmat * a_vert;
v_texcoord=vec2(u_texmat*vec4(a_texcoord));
v_color=a_color/255.0;
}

[FRAGMENT]

#include "common.glsl"

IN mediump vec2 v_texcoord;
IN mediump vec4 v_color;

void main()
{ //u_fogstart=time   u_fogend=size
lowp vec4 c=v_color;

mediump float r=length(v_texcoord);
mediump float wave = 1.0 - clamp(abs(1.5-mod((r/u_fogend+u_fogstart)*4.0,3.0)), 0.0, 1.0);

c*=texture2D(u_tex,v_texcoord*((10.0-wave)/10.0));

c.g *= 0.5 + 0.5*wave;
c.g += 0.4*clamp(1.0-r/u_fogend,0.0,1.0)+0.2*wave;
c.g *= u_fogcolor.a;

gl_FragColor=c;
}
