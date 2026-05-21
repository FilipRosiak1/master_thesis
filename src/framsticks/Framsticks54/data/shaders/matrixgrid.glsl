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
{
lowp vec4 c=v_color;

mediump float xphase=sin(u_fogstart);
mediump float yphase=cos(u_fogstart);
mediump float xwave = 0.5 - clamp(abs(1.5-mod(v_texcoord.x*0.1+xphase*4.0,3.0)), 0.0, 0.5);
mediump float ywave = 0.5 - clamp(abs(1.5-mod(v_texcoord.y*0.1+yphase*4.0,3.0)), 0.0, 0.5);
c*=texture2D(u_tex,v_texcoord+vec2(xwave*yphase,-ywave*xphase)*0.4);
c.g *= 0.5 + max(pow(abs(yphase),3.0)*xwave,pow(abs(xphase),3.0)*ywave);
c.g *= u_fogcolor.a;

gl_FragColor=c;
}
