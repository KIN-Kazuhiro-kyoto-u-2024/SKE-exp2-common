# Simulator
`simulator/acrobot.py`のinitialize_episode関数を変更することで初期値を設定できる。（デフォルトではelbowの角度が$`[-0.5\pi, 0.5\pi]`$の間のランダムな値で、shouderの角度が$`0`$）

`simulator/acrobot.xml`を変えることで、物体の重さ、長さ、ジョイントのダンピングや摩擦を変更できる。

具体的には、下の表に対応する数値を下のコードの対応する位置に入れる。

| パラメータ | upper arm | lower arm |
| :---: | :---: | :---: |
| mass | m_u | m_l |
| length | l_u | l_l |
| size(diameter) | s_u | s_l |
| damping | d_u | d_l |
| friction(静止摩擦) | f_u | f_l |

```xml
<body name="upper_arm" pos="0 0 2">
    <joint name="shoulder" type="hinge" axis="0 0 1" damping="d_u" frictionloss="f_u"/>
    <geom name="upper_arm_decoration" material="decoration" type="cylinder" fromto="0 0 -.01 0 0 .01" size="s_u" mass="0"/>
    <geom name="upper_arm" fromto="0 0 0 l_u 0 0" size="s_u" material="self" rgba="1 1 0 0.6" mass="m_u"/>
    <body name="lower_arm" pos="l_u 0 0">
        <joint name="elbow" type="hinge" axis="1 0 0" damping="d_l" frictionloss="f_l"/>
        <geom name="lower_arm" fromto="0 0 0 0 0 l_l" size="s_l" material="self" rgba="1 0 1 0.6" mass="m_l"/>
        <site name="tip" pos="0 0 -l_l" size="0.01"/>
        <geom name="lower_arm_decoration" material="decoration" type="cylinder" fromto="-.01 0 0 .01 0 0" size="s_l" mass="0"/>
    </body>
</body>
```