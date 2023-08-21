from django.http import JsonResponse
import os
import sys
import hashlib
import json
from django.db import IntegrityError
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets
import operator
import shutil
import requests
from PIL import Image, ImageOps
from pathlib import Path
from PIL import Image
from collections import Counter
import base64
from io import BytesIO
from django.views.decorators.csrf import csrf_exempt
import secrets
import pandas as pd
import torch
from api_face_reco.models import Tensor, Person
from django.conf import settings
import cv2
from .log import Logger
import glob

if 'log_views' not in globals():
    log_views = Logger('views', level=logging.ERROR).run()

def format_dataframe_from_database(df):
    # get db
    df['encodedface'] = df['encodedface'].apply(lambda x: torch.Tensor(x))
    return list(df['encodedface']), list(df['filename'])

def format_dataframe_to_database(df):
    #transform to a list of float accepted by the postgresql db
    df['encodedface'] = df['encodedface'].apply(lambda x: x.tolist())
    return df

def extract_picture (video, client_id, filename):
    path = video[:-4]
    path_pictures = settings.MEDIA_ROOT + '/pictures_users/' + client_id
    if os.path.exists(path):
        for file in os.listdir(path):
            os.remove(path + '/' + file)
        os.removedirs(path)
    os.mkdir(path)
    cap = cv2.VideoCapture(video)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # frame=cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
        if not ret:
            break

        if i % 2 == 0:
            cv2.imwrite(
                path_pictures + '/' + filename + '_' + str(
                    i) + '.jpg', frame)
        i += 1
    cap.release()
    cv2.destroyAllWindows()


def collate_fn(x):
    """This function is a silly function for the DataLoader"""
    return x[0]


def embeddings_db(aligned, old_batch, batch_size, device, size_of_db, resnet):
    """This function is to send after the mtcnn batches in the resnet

    :param aligned: this is the output of the mtcnn

    :param old_batch: int this number is initialize to 0 and help to keep the count of the batch that have been process

    :param batch_size: int size of the batch size usually 16

    :param device: str specify on which device the algorithm works ('cpu' or 'cuda:0')

    :param size_of_db: int total numbers of faces to encode

     :param resnet: resnet cnn

    :return: output batch of tensors after resnet
    """
    embeddings_all = []
    for j in range(size_of_db // batch_size):
        faces_batch = []
        for i in range(old_batch, batch_size + old_batch):
            faces_batch.append(aligned[i])
        old_batch = batch_size + old_batch
        faces_batch = torch.stack(faces_batch).to(device)
        embeddings = resnet(faces_batch).detach().cpu()
        embeddings_all.append(embeddings)
    if (size_of_db % batch_size) != 0:
        rest = size_of_db % batch_size
        for i in range(old_batch, rest + old_batch):
            faces_batch = []
            faces_batch.append(aligned[i])
            old_batch = rest + old_batch
            faces_batch = torch.stack(faces_batch).to(device)
            embeddings = resnet(faces_batch).detach().cpu()
            embeddings_all.append(embeddings)

    return embeddings_all


def getKNeighbors(X, y, image_to_test, k):
    """

    This function calculate the distance between an encoding face test_image and the encoding face database.
    Then it select the k closest encoding face.

    :param X: list of all the encoding face database (output of facetensordatabase() function)
    :param y: list of the name corresponding to these encoding face (output of facetensordatabase() function)
    :param image_to_test: encoding face to test against the database
    :param k: numbers of closest neighbours we want to use for the KNN

    :return: returns 1 variables: A list containing tuple organised as follow: (face_encoding tensor, distance, name)

    """
    X = torch.stack(X)
    dist = ((image_to_test - X) ** 2)
    dist = dist.sum(axis=1)
    dist = torch.sqrt(dist)

    value, indices = torch.sort(dist)
    close_neighbors = []
    top_ten = []
    for x in range(k):
        close_neighbors.append((X[indices[x]], value[x], y[indices[x]]))
        top_ten.append(y[indices[x]])
    for top in range(100 - k):
        top_ten.append(y[indices[top]])
    l_sorted = Counter(top_ten).most_common()
    l_sorted = [i[0] for i in l_sorted]
    return close_neighbors, l_sorted


def getResponse(close_neighbors, distance_threshold, distance_min):
    """

    This function is used to classify the unknown encoding face to a corresponding member of the database when the
    distance is close enough.

    :param close_neighbors: output of getKNeighbors() function
    :param distance_threshold: if the distance between the unknown encoding face and the closest neighbour is >0.6 than
                      the unknown encoding face is classify as unknown

    :param distance_min: if the distance between the unknown encoding face and the closest neighbour is <0.2 than
                      the unknown encoding face is directly classify as this person

    :return: returns 2 variables: * sortedVotes[0][0]: A list containing the name getting the majority of vote.
                                  * are_matches: A list of Bol saying if the is a match or not with database
                                              True=yes
                                              False=No

    """
    classVotes = {}
    classVotes["Unknown Person"] = 0
    if distance_min > close_neighbors[0][1]:
        are_matches = True
        return close_neighbors[0][2], are_matches
    else:
        for x in close_neighbors:
            if distance_min <= x[1] <= distance_threshold:
                if x[2] in classVotes:
                    classVotes[x[2]] += 1
                else:
                    classVotes[x[2]] = 1
            else:
                classVotes["Unknown Person"] += 1

        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

        if len(sortedVotes) > 1:
            are_matches = (sortedVotes[0][0] != "Unknown Person") and (sortedVotes[0][1] > sortedVotes[1][1])
        else:
            are_matches = (sortedVotes[0][0] != "Unknown Person")
        return sortedVotes[0][0], are_matches


def who_ru(img_64, mtcnn, resnet, device=None, embeddings_db=None, names=None, distance_min=0.3,
           distance_threshold=0.7, k=3):
    """
    Recognizes faces from a base64image image using a KNN classifier and return an Identity or an unknown person

    :param img_64: base64 images

    :param mtcnn: mtcnn model for faces identification in the picture

    :param resnet: resnet model to get the encode face embeddings

    :param device: str specify on which device the algorithm works ('cpu' or 'cuda:0')

    :param embeddings_db : encode faces database

    :param names : ID client for eech encoding faces

    :param distance_min: if the distance between the unknown encoding face and the closest neighbour is <0.3 than
                      the unknown encoding face is directly classify as this person

    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
        of mis-classifying an unknown person as a known one.

    :param k: numbers of closest neighbours we want to use for the KNN


    :return: return an Identity or an unknown person
    """

    im_bytes = base64.b64decode(img_64)  # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
    img = Image.open(im_file)

    # Load image file and find face locations
    name = 'None'
    top10 = 'None'

    try:
        x_aligned, prob = mtcnn(img, return_prob=True)
    except:
        return f'{name}'

    if x_aligned is not None:
        if prob > 0.6:

            result = x_aligned.unsqueeze(0).to(device)
            predict = resnet(result).detach().cpu()

            # See if the face is a match for the known face(s)
            for face in predict:
                neighbors, top10 = getKNeighbors(embeddings_db, names, face, k)
                predi, are_matches = getResponse(neighbors, distance_threshold, distance_min)
                name = "unknown"
                if are_matches and predi != 'Unknown':
                    name = predi
                else:
                    return f'{name}'

    return f'{name}/{top10}'


@csrf_exempt
def index(request):
    image = None
    if request.method == 'POST':
        image = request.POST.get('imgb64')

    if image is None:
        return JsonResponse('None', safe=False)

    if "client_token" in request.headers:
        identity = request.headers["client_token"]

        queryset = Tensor.objects.filter(person__identity_server_group=identity)

        embeddings = [torch.Tensor(t.encodedface) for t in queryset]
        names = [t.person.token for t in queryset]

        # define the device:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # define the model MTCNN:
        mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709,
                    post_process=True, device=device)

        # define the model:
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        return JsonResponse(who_ru(image, mtcnn=mtcnn, resnet=resnet, device=device, embeddings_db=embeddings,
                                names=names, distance_min=0.3, distance_threshold=0.8, k=3), safe=False)

@csrf_exempt
def url_test(request, resp):
    if request.method == "POST":
        return JsonResponse(f'{resp} received in POST', safe=False)
    return JsonResponse(f'{resp} received in GET', safe=False)


@csrf_exempt
def new_entries(request):
    video_binary_string = None
    identity = None

    if request.method == 'POST' and "client_token" in request.headers:
        video_binary_string = request.POST.get('vdo_rec')
        identity = request.headers["client_token"]
    if video_binary_string or identity is None :
        return JsonResponse('None', safe=False)

    # set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # log the devive
    log_views.info(f'Running on device: {device}')

    # load the mtcnn network and the resnet network
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    # define workers
    workers = 0 if os.name == 'nt' else 4

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    decoded_string = base64.b64decode(video_binary_string)
    filename = request.POST.get('filename')
    log_views.info(f'filename: {filename}')
    path_main = settings.MEDIA_ROOT + '/pictures_users/' + identity + "/"
    path_video = settings.MEDIA_ROOT + '/videos_users/' + identity + "/" + filename + '/'
    path_pictures = settings.MEDIA_ROOT + '/pictures_users/' + identity + "/" + filename + '/'
    path_profile_pic = settings.MEDIA_ROOT + '/profile_pic_users/' + identity + "/" + filename + '/'
    log_views.info(f'path: {path_video}')
    try:
        Path(path_video).mkdir(parents=True, exist_ok=True)
        path_video_named = path_video + '.mp4'
        if os.makedirs(os.path.dirname(path_video_named), exist_ok=True):
            shutil.rmtree(path_video_named)
            os.makedirs(os.path.dirname(path_video_named))
        with open(path_video_named, 'wb') as wfile:
            wfile.write(decoded_string)
        extract_picture(path_video_named, identity, filename)

    except Exception as prob:
        log_views.info(f'extract pictures failed: {prob}')

    classified_list = []
    desired_size = 200
    parse_photo = glob.glob(path_pictures + '*.jpg')
    if len(parse_photo) > 0:
        face = False
        while not face:
            for picture in parse_photo:
                log_views.info(f'picture: {picture}')
                img = Image.open(picture)
                width, height = img.size
                pict_center = torch.as_tensor([width / 2, height / 2])
                boxes, b, landmarks = mtcnn.detect(img, landmarks=True)
                if boxes is not None:
                    box = boxes[0]
                    landmark = landmarks[0]
                    if landmark.shape == (5, 2):
                        nose = torch.as_tensor([landmark[2, 0], landmark[2, 1]])
                        dist_center = ((pict_center - nose) ** 2)
                        dist_center = torch.sqrt(dist_center.sum())
                        dist = abs(landmark[3, 0] - landmark[4, 0])
                        classified_list.append((picture, dist, dist_center, box))

            if len(classified_list) != 0:  # sort according to the min distance nose-center_pic
                classified_list.sort(key=lambda x: x[2])

                # select the 10% smallest distance distance
                best_pictures_list = classified_list[:int(len(classified_list) / 10)]

                # sort according to the max distance left_mouth-right_mouth
                best_pictures_list.sort(key=lambda x: x[1], reverse=True)

                for profile in best_pictures_list:
                    img_profile = Image.open(profile[0])

                    if int(profile[3][0]) > -20 and int(profile[3][1]) > -20 and int(profile[3][2]) < \
                            img_profile.size[0] + 20 and int(profile[3][1]) < img_profile.size[1] + 20:
                        left = round(int(profile[3][0] - ((1 * (profile[3][3] - profile[3][1])) / 2)))
                        if left <= -20:
                            left = 0

                        top = round(int(profile[3][1] - ((1 * (profile[3][2] - profile[3][0])) / 2)))
                        if top <= -20:
                            top = 0

                        right = round(int(profile[3][2] + ((1 * (profile[3][3] - profile[3][1])) / 2)))
                        if right >= img_profile.size[0] + 20:
                            right = img_profile.size[0]

                        bottom = round(int(profile[3][3] + ((1 * (profile[3][2] - profile[3][0])) / 2)))
                        if bottom >= img_profile.size[1] + 20:
                            bottom = img_profile.size[1]

                        img_copy = img_profile.copy()
                        face_cropped = img_copy.crop((left, top, right, bottom))
                        old_size = face_cropped.size  # old_size[0] is in (width, height) format
                        #print(old_size)
                        if min(old_size) < 200:
                            continue

                        ratio = float(desired_size) / max(old_size)
                        new_size = tuple([int(x * ratio) for x in old_size])

                        delta_w = desired_size - new_size[0]
                        delta_h = desired_size - new_size[1]
                        padding = (
                        delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                        im = face_cropped.resize(new_size, Image.ANTIALIAS)
                        new_im = ImageOps.expand(im, padding)
                        Path(path_profile_pic).mkdir(parents=True, exist_ok=True)
                        new_im.save(path_profile_pic + 'profile_pic.jpg')
                        log_views.info(f'profil pic found and saved')
                        face = True
                        break

            if not face:
                #need to raise a warning if o faces have been found like a requests.get(path_request)
                break

        if face:
            #async_to_sync(connect_test)(path_dest=path_profile_pic, profile=False, websocket_url=websocket_url)

            os.remove(path_profile_pic + 'profile_pic.jpg')
            os.rmdir(path_profile_pic)

            # create a loader from the folder containing all the new entries
            dataset = datasets.ImageFolder(path_main)
            dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
            loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

            aligned = []
            names = []
            for x, y in loader:
                try:
                    x_aligned, prob = mtcnn(x, return_prob=True)
                    if x_aligned is not None:
                        if prob >= 0.98:
                            log_views.info(f'Face detected with probability: {prob:.4f}')
                            aligned.append(x_aligned)
                            names.append(dataset.idx_to_class[y])
                except:
                    log_views.info(f'no face detected')

            log_views.info(f'numbers of encoded faces before resnet: {len(aligned)}'
                        f', number of encoded names before resnet: {len(names)}')

            # batch pass mtcnn result into resnet

            embeddings_all = embeddings_db(aligned=aligned, old_batch=0, batch_size=16, device=device,
                                                   size_of_db=len(aligned), resnet=resnet)

            # flattening embeddings
            flat_embeddings = []
            for sublist in embeddings_all:
                for item in sublist:
                    flat_embeddings.append(item)

            log_views.info(f'numbers of encoded faces after resnet: {len(flat_embeddings)}'
                            f', number of encoded names after resnet: {len(names)}')

            number_faces = {name: names.count(name) for name in names}

            for key in number_faces:
                if int(number_faces[key]) < 50:
                    #need to raise a warning if the number of faces is below 50
                    pass

                # create the dataframe with new entries
            list_person = [{'person': x,
                            'encodedface': y.tolist(),
                            'md5_hash': hashlib.md5(json.dumps(y, sort_keys=True).encode('utf-8')).hexdigest()}
                           for x, y in zip(names, flat_embeddings)
                           ]

            for dict_person in list_person:
                obj, created = Person.objects.create(token=dict_person['person'], identity_server_group=identity)
                t = Tensor(person=obj, encodedface=dict_person['encodedface'], md5hash=dict_person['md5_hash'])
                try:
                    t.save()
                except IntegrityError:
                    #need a correct warning at this stage
                    pass

            # erased pictures
            for pict in parse_photo:
                os.remove(pict)

            log_views.info(f'faces successfully incorporated in the database')

    return JsonResponse("Succefully Incorporated",safe=False)



@csrf_exempt
def url_check(request):
    JsonResponse("Hello", safe=False)
