### Faceshifter tornado server
Implementation and deployment of https://arxiv.org/abs/1912.13457.

### Requirements
Python 3.6+

## Step-by-step installation
1. Install MongoDB [Guide](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/)
2. Create mongo database with name `faceshifter`: in shell run `mongo` and then `use faceshifter;`
2. `cd faceshifter_tornado/`
3. `python3 -m venv ENV/`
4. `pip install -r requirements.txt`
5. Download weights for AEI generator [google drive link](https://drive.google.com/file/d/1z1htsPJi-hfTcD8akOO3xJNFy9jYccGc/view?usp=sharing)
6. Extract weights  `tar -xvf faceshifter_weights.tar -C faceshifter/model_weights/`
7. Run tests: `python -m tornado.test.runtests web.tests`
8. Launch server with inference worker `python app.py`

## Try it
1. Send photos for swapping. Make `POST` request on `/upload_image` with form-data files: `source, target`. Copy `id` from response.
2. Get swapping results. Make `GET` request to `/result/$id/`. 
3. Check `result_url`

### Deployment notes and todos:
- Add highload testing with [locust.io](https://locust.io/)
- Use PyTorch jit compiled graph for saving RAM and speeding up inference
- Docker for faster deployment
- Use PostgreSQL or another relational database with migrations, transaction functionality for better data consistency
- Try to use JIT (PyPy) for faster python code execution


Model implementation reference: https://github.com/Heonozis/FaceShifter-pytorch/
