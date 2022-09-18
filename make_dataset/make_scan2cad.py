import os
import shutil
import numpy as np
from tqdm import tqdm
import json
import open3d as o3d
import quaternion
import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M


def calc_Mbbox(model):
    trs_obj = model["trs"]
    bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
    center_obj = np.asarray(model["center"], dtype=np.float64)
    trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
    rot_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
    q_obj = np.quaternion(rot_obj[0], rot_obj[1], rot_obj[2], rot_obj[3])
    scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

    tcenter1 = np.eye(4)
    tcenter1[0:3, 3] = center_obj
    trans1 = np.eye(4)
    trans1[0:3, 3] = trans_obj
    rot1 = np.eye(4)
    rot1[0:3, 0:3] = quaternion.as_rotation_matrix(q_obj)
    scale1 = np.eye(4)
    scale1[0:3, 0:3] = np.diag(scale_obj)
    bbox1 = np.eye(4)
    bbox1[0:3, 0:3] = np.diag(bbox_obj)
    M = trans1.dot(rot1).dot(scale1).dot(tcenter1).dot(bbox1)
    return M


def vanish(Mbbox, scan_warped):
    Mbbox_inverse = np.linalg.inv(Mbbox)
    scan_warped_warped = np.dot(Mbbox_inverse, scan_warped.T).T[:, :3]
    idx = ((np.multiply((scan_warped_warped < 1.1), (scan_warped_warped > -1.1))).sum(-1) < 3).nonzero()[0]

    return idx


def show_dataset(trans,scan_and_cad,cad_origin):
    cad_homo = np.concatenate((cad_origin, np.ones([cad_origin.shape[0], 1])), axis=-1)
    print(len(trans))
    for i in range(len(trans)):
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(scan_and_cad - 0.01)
        pcd0.paint_uniform_color([1,0,0])

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(cad_origin)
        pcd1.paint_uniform_color([0,0,0])

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(np.matmul(trans[i],cad_homo.T).T[:,:3])
        pcd2.paint_uniform_color([0,0,1])


        o3d.visualization.draw_geometries([pcd0, pcd1, pcd2])



DATASET = add_argument_group('Make dataset')
DATASET.add_argument('--scan2cad', type=str, default='/media/ymz/软件/PointCLM/scan2cad_download_link/full_annotations.json')
DATASET.add_argument('--scannet', type=str, default='/media/ymz/软件/PointCLM/scans/')
DATASET.add_argument('--shapenet', type=str, default='/media/ymz/软件/PointCLM/shapenet/')
DATASET.add_argument('--output', type=str, default='/media/ymz/软件/PointCLM/scan2cad/')



if __name__ == '__main__':
    args = parser.parse_args()
    root = args.scan2cad
    output_root = args.output


    with open(root, encoding='utf-8') as f:
        line = f.readline()
        d = json.loads(line)
        print(type(d))
        f.close()

#collect data for making dataset
    dataset = []

    for r in tqdm(d):
        id_scan = r['id_scan']
        trans_scan = r['trs']

        cat_cad = {}

        for i, model in enumerate(r['aligned_models']):

            id_cad = model['id_cad']
            catid_cad = model['catid_cad']
            id_cat = id_cad + catid_cad

            if id_cat in cat_cad:
                cat_cad[id_cat].append(i)
            else:
                cat_cad[id_cat] = [i]

        for item in cat_cad:
            if len(cat_cad[item]) > 1:
                data = {}
                data['id_scan'] = id_scan
                data['trans_scan'] = trans_scan
                data['cad'] = []

                for i in cat_cad[item]:
                    data['cad'].append(r['aligned_models'][i])

                dataset.append(data)

    np.save('scan2cad.npy', np.array(dataset))

#begin to make dataset
    #
    count = 0

    for item in tqdm(dataset):
        id_scan = item['id_scan']
        trans_scan = item['trans_scan']
        scan_root = args.scannet + id_scan + '/' + id_scan + '_vh_clean_2.ply'
        pcd = o3d.io.read_point_cloud(scan_root)

        Mscan = make_M_from_tqs(trans_scan['translation'], trans_scan['rotation'], trans_scan['scale'])
        scan_homo = np.concatenate((np.array(pcd.points), np.ones([np.array(pcd.points).shape[0], 1])), axis=-1)
        scan_warped = np.dot(Mscan, scan_homo.T).T

        scale_min = []
        for cad in item['cad']:
            scale_min.append(cad['trs']['scale'])
            id_cad = cad["id_cad"]
            catid_cad = cad["catid_cad"]

        scale_min = np.array(scale_min).min(0).tolist()
        cadroot = args.shapenet + catid_cad + '/' + id_cad + '/model_normalized.obj'
        cad = o3d.io.read_triangle_mesh(cadroot)
        cad = cad.sample_points_uniformly(10000)
        cad_homo = np.concatenate((np.array(cad.points), np.ones([np.array(cad.points).shape[0], 1])), axis=-1)

        T = np.eye(4)
        R = np.eye(4)
        S = np.eye(4)
        S[0:3, 0:3] = np.diag(scale_min)

        cad_origin = np.matmul(T.dot(R).dot(S), cad_homo.T).T[:, :3]
        trans = []

        for model in item['cad']:
            t = model["trs"]["translation"]
            q = model["trs"]["rotation"]
            s = [1, 1, 1]

            Mcad = make_M_from_tqs(t, q, s)
            Mcad_min = make_M_from_tqs(t, q, scale_min)
            Mbbox = calc_Mbbox(model)
            trans.append(Mcad)

            idx = vanish(Mbbox, scan_warped)

            cad_warped = np.matmul(Mcad_min, cad_homo.T).T
            scan_part = scan_warped[idx]
            scan_warped = np.concatenate((scan_part, cad_warped), axis=0)

        scan_and_cad = scan_warped[:, :3]
        np.random.shuffle(scan_and_cad)
        trans = np.array(trans)

        savename = output_root + 'data{:05d}.npz'.format(count)
        np.savez(savename, scan=scan_and_cad, shape=cad_origin, trans=trans)
        count = count + 1