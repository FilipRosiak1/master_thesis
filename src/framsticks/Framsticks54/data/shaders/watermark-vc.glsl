glcompat_attrs=6 //texture+vertcolors

[VERTEX]

#include "common.glsl"

IN vec4 a_vert;
IN vec4 a_texcoord;
IN vec4 a_color;
OUT mediump vec4 v_color;
OUT mediump vec2 v_texcoord,v_texcoord2;

void main()
{
    gl_Position = u_projmodmat * a_vert;
    v_texcoord=vec2(u_texmat*vec4(a_texcoord));
    v_texcoord2=vec2(u_texmat2*a_vert);
    v_color=a_color/255.0;
}

[FRAGMENT]

#include "common.glsl"

IN mediump vec4 v_color;
IN mediump vec2 v_texcoord,v_texcoord2;

void main()
{
    mediump vec4 c=v_color;
    c*=texture2D(u_tex,v_texcoord)*texture2D(u_tex2,v_texcoord2);
    gl_FragColor=c;
}
