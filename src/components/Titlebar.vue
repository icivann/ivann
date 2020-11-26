<template>
  <div class="titlebar">
    <div class="logo">
      <img class="img-fluid titlebar-logo mr-2" src="@/assets/images/nn_logo.png" alt="IVANN"/>
      <span class="text">IVANN</span>
    </div>
    <div class="buttons">
      <a class="icon-button" href="https://github.com/icivann/ivann" target="_blank" title="GitHub">
        <i class="titlebar-icon fab fa-github fa-lg mx-2"/>
      </a>
      <span class="icon-button" @click="codegen" title="Generate Code">
        <i class="titlebar-icon fas fa-code fa-lg mx-2"/>
      </span>
      <input
        type="file"
        id="upload-file"
        style="display: none"
        @change="load"
      >
      <span class="icon-button" @click="uploadFile" title="Load Project">
        <i class="titlebar-icon fas fa-folder-open fa-lg mx-2"/>
      </span>
      <span class="icon-button" @click="save" title="Save Project">
        <i class="titlebar-icon fas fa-save fa-lg mx-2"/>
      </span>
      <span class="icon-button" @click="newProject" title="New Project">
        <i class="titlebar-icon fas fa-file fa-lg mx-2"/>
      </span>
    </div>
<!--  Modals  -->
    <Modal v-model="modalOpen" header="Export">
      <ExportModal/>
    </Modal>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Getter, Mutation } from 'vuex-class';
import { EditorModel, EditorModels } from '@/store/editors/types';
import { download } from '@/file/Utils';
import {
  FILENAME,
  Save,
  saveEditor,
  saveEditors,
  SaveWithNames,
} from '@/file/EditorAsJson';
import { FilenamesList, ParsedFile } from '@/store/codeVault/types';
import EditorType from '@/EditorType';
import Modal from '@/components/modals/Modal.vue';
import ExportModal from '@/components/modals/ExportModal.vue';

@Component({
  components: { ExportModal, Modal },
})
export default class Titlebar extends Vue {
  @Getter('currEditorType') currEditorType!: EditorType;
  @Getter('allEditorModels') editorModels!: EditorModels;
  @Getter('currEditorModel') overviewEditor!: EditorModel;
  @Getter('currEditorModel') currEditor!: EditorModel;
  @Getter('saveWithNames') saveWithNames!: SaveWithNames;
  @Getter('files') files!: ParsedFile[];
  @Mutation('loadEditors') loadEditors!: (save: Save) => void;
  @Mutation('loadFiles') loadFiles!: (files: ParsedFile[]) => void;
  @Mutation('resetState') resetState!: () => void;
  @Getter('filenamesList') filenamesList!: FilenamesList;
  @Getter('file') file!: (filename: string) => ParsedFile;

  private modalOpen = false;

  private codegen() {
    this.modalOpen = true;
  }

  // Trigger click of input tag for uploading file
  private uploadFile = () => {
    const element = document.getElementById('upload-file');
    if (!element) return;
    element.click();
  }

  private load() {
    const { files } = document.getElementById('upload-file') as HTMLInputElement;
    if (!files) return;

    const fr: FileReader = new FileReader();
    fr.onload = (event) => {
      if (!event.target) return;

      // Load all editors using parsed file and set Local Storage.
      const parsed = JSON.parse(event.target.result as string);
      this.loadEditors(parsed.editors);
      this.loadFiles(parsed.files);

      // Clear except `cookie:accepted`
      const cookieAccepted = localStorage.get('cookie:accepted');
      localStorage.clear();
      localStorage.setItem('cookie:accepted', cookieAccepted);

      // Save new project to Local Storage
      const { saveWithNames } = this;
      localStorage.setItem('unsaved-project', JSON.stringify(saveWithNames));
      localStorage.setItem('unsaved-editor-Overview', JSON.stringify(saveEditor(this.overviewEditor)));
      saveWithNames.modelEditors.forEach((name) => {
        const modelEditor = this.editorModels.modelEditors.find((editor) => editor.name === name);
        if (modelEditor) {
          localStorage.setItem(`unsaved-editor-${name}`, JSON.stringify(saveEditor(modelEditor)));
        }
      });
      saveWithNames.dataEditors.forEach((name) => {
        const dataEditor = this.editorModels.dataEditors.find((editor) => editor.name === name);
        if (dataEditor) {
          localStorage.setItem(`unsaved-editor-${name}`, JSON.stringify(saveEditor(dataEditor)));
        }
      });

      // Save Code Vault
      const { filenamesList } = this;
      localStorage.setItem('unsaved-code-vault', JSON.stringify(filenamesList));
      const files = filenamesList.filenames.map((filename: string) => JSON.parse(localStorage.getItem(`unsaved-file-${filename}`)!));
      filenamesList.filenames.forEach((filename) => {
        localStorage.setItem(`unsaved-file-${filename}`, JSON.stringify(this.file(filename)));
      });
    };

    // Trigger the file to be read
    fr.readAsText(files[0]);
  }

  private save() {
    const {
      overviewEditor,
      modelEditors,
      dataEditors,
    } = this.editorModels;

    const editorsSaved: Save = {
      overviewEditor: saveEditor(overviewEditor),
      modelEditors: saveEditors(modelEditors),
      dataEditors: saveEditors(dataEditors),
    };

    download(FILENAME, JSON.stringify({
      editors: editorsSaved,
      files: this.files,
    }));
  }

  private newProject() {
    // TODO FE-55 Implement better Confirm dialog
    if (window.confirm('Are you sure you want to create a new project? '
      + 'All unsaved progress will be lost.')) {
      this.resetState();
      // Clear except `cookie:accepted`
      const cookieAccepted = localStorage.get('cookie:accepted');
      localStorage.clear();
      localStorage.setItem('cookie:accepted', cookieAccepted);

      localStorage.setItem('unsaved-project', JSON.stringify(this.saveWithNames));
      localStorage.setItem('unsaved-editor-Overview', JSON.stringify(saveEditor(this.editorModels.overviewEditor)));
    }
  }
}
</script>

<style lang="scss" scoped>
  .titlebar {
    height: 2.5rem;
    background-color: var(--background-alt);
    border-bottom: 1px solid var(--grey);
    margin-right: -15px;
    margin-left: -15px;
  }

  .titlebar-logo {
    height: 1.2rem;
  }

  .text {
    color: var(--foreground);
  }

  .titlebar-icon {
    color: var(--foreground);
  }

  .icon-button {
    background-color: var(--background-alt);
    margin: 0.15rem;
    padding: 0.3rem 0.1rem;

    &:hover {
      background-color: #2c2c2c;
      cursor: pointer;
    }
  }

  .buttons {
    float: right;
    display: flex;
  }

  .logo {
    margin-top: 0.4rem;
    margin-left: 0.5rem;
    float: left;
    user-select: none;
  }
</style>
