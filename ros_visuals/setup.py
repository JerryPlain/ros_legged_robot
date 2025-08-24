from setuptools import find_packages, setup

package_name = 'ros_visuals'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/ros_visuals/launch', ['launch/launch_t11.py', 'launch/launch_t12.py', 'launch/launch_t13.py', 'launch/talos_rviz.launch.py']),
        ('share/ros_visuals/config', ['config/t11.rviz', 'config/t12.rviz', 'config/t13.rviz', 'config/talos.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='devel',
    maintainer_email='devel@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            't11 = ros_visuals.t11:main',
            't12 = ros_visuals.t12:main',
            't13 = ros_visuals.t13:main',
            'teleop_marker = ros_visuals.teleoperation:main',
            'interactive = ros_visuals.interactive_target:main',
            't4_standing = ros_visuals.t4_standing:main',
            'one_leg_stand = ros_visuals.one_leg_stand:main',
            'squating = ros_visuals.squating:main',
            't51 = ros_visuals.t51:main',
            't52 = ros_visuals.t52:main',
        ],
    },
)