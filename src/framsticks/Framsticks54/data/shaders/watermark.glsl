glcompat_attrs=4 //texture

[VERTEX]

#include "common.glsl"

IN vec4 a_vert;
IN vec4 a_texcoord;
OUT mediump vec2 v_texcoord,v_texcoord2;

void main()
{
    gl_Position = u_projmodmat * a_vert;
    v_texcoord=vec2(u_texmat*vec4(a_texcoord));
    v_texcoord2=vec2(u_texmat2*a_vert);
}

[FRAGMENT]

#include "common.glsl"

IN mediump vec2 v_texcoord,v_texcoord2;

void main()
{
    mediump vec4 c=u_color;
    c*=texture2D(u_tex,v_texcoord)*texture2D(u_tex2,v_texcoord2);
    gl_FragColor=c;
}
