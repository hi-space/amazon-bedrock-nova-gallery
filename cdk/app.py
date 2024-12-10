#!/usr/bin/env python3
import os

import aws_cdk as cdk

from cdk.cdk_stack import CdkStack


app = cdk.App()
app_name = "nova-gallery"
CdkStack(app, f"{app_name}-stack", app_name=app_name)

app.synth()
