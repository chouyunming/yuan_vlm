from setuptools import find_packages, setup

package_name = 'drone_vlm'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anderson',
    maintainer_email='ander0315@gmail.com',
    description='drone_vlm: Gemma VLM over llama-server. vlm_test = ROS-free '
                'video/image benchmark; vlm_node = ROS image-topic.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlm_node = drone_vlm.vlm_node:main',
        ],
    },
)
