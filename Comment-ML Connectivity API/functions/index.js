const functions = require('firebase-functions');
const admin = require('firebase-admin');

admin.initializeApp(functions.config().firebase);

exports.processNewPost = functions.database.ref('/posts/{postId}')
    .onCreate(async (snapshot, context) => {
        const postData = snapshot.val();
        const postText = postData.text;
        
        console.log('New post:', postText);
        
        // Lakukan operasi atau aksi lain sesuai kebutuhan Anda dengan nilai teks dari posting ini.
        // Misalnya, mengirim notifikasi, memproses data, atau menyimpan data ke penyimpanan lain.

        return null; // Tambahkan nilai balik sesuai kebutuhan Anda
    });