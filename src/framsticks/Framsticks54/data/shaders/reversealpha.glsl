glcompat_attrs=4 //texture

[VERTEX]

#include "common.glsl"

IN vec4 a_vert;
IN vec4 a_texcoord;
OUT mediump vec2 v_texcoord;

void main()
{
    gl_Position = u_projmodmat * a_vert;
    v_texcoord=vec2(u_texmat*vec4(a_texcoord));
}

[FRAGMENT]

#include "common.glsl"

IN mediump vec2 v_texcoord;

void main()
{
    mediump vec4 c=u_color;
    lowp vec4 t=texture2D(u_tex,v_texcoord);
    c*=vec4(t.rgb,1.0-t.a);
    gl_FragColor=c;
}
