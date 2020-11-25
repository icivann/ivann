<template>
  <div class="left d-flex flex-column h-100 text-center">
    <!-- Build -->
    <div
      class="build tab-button"
      :class="isSelected(editorType.OVERVIEW)"
      @click="switchOverviewEditor"
    >
      <i class="fas fa-hammer tab-icon"/>
    </div>

    <div class="py-1 px-2">
      <hr/>
    </div>

    <!-- Model -->
    <div
      class="model tab-button"
      :class="isSelected(editorType.MODEL)"
      @mouseover="displayNavbarContextualMenu(editorType.MODEL)"
      @mouseleave="hideNavbarContextualMenu()"
    >
      <img class="navbar-logo tab-icon" src="@/assets/images/nn_logo.png" alt="IVANN"/>
      <NavbarContextualMenu
        class="navbar-contextual-menu"
        v-if="isModelContextualMenuOpen"
        :editors="modelEditors"
        :editor-type="editorType.MODEL"
      />
    </div>

    <div class="py-1 px-2">
      <hr/>
    </div>

    <!-- Data -->
    <div
      class="data tab-button"
      :class="isSelected(editorType.DATA)"
      @mouseover="displayNavbarContextualMenu(editorType.DATA)"
      @mouseleave="hideNavbarContextualMenu()"
    >
      <i class="fas fa-database tab-icon"/>
      <NavbarContextualMenu
        class="navbar-contextual-menu"
        v-if="isDataContextualMenuOpen"
        :editors="dataEditors"
        :editor-type="editorType.DATA"
      />
    </div>

    <!-- Code Vault -->
    <div class="flex-grow-1"/>
    <div
      class="build tab-button"
      :class="inCodeVault && 'selected'"
      @click="enterCodeVault"
    >
      <i class="fab fa-python tab-icon"/>
    </div>
  </div>
</template>

<script lang="ts">
import { Vue, Component } from 'vue-property-decorator';
import EditorType from '@/EditorType';
import { mapGetters, mapMutations } from 'vuex';
import { Getter, Mutation } from 'vuex-class';
import NavbarContextualMenu from '@/components/navbar/NavbarContextualMenu.vue';
import { EditorSave, saveEditor } from '@/file/EditorAsJson';
import { EditorModel } from '@/store/editors/types';

@Component({
  components: { NavbarContextualMenu },
  computed: mapGetters([
    'modelEditors',
    'dataEditors',
    'inCodeVault',
  ]),
  methods: mapMutations(['enterCodeVault']),
})
export default class Navbar extends Vue {
  private editorType = EditorType;
  private isModelContextualMenuOpen = false;
  private isDataContextualMenuOpen = false;
  @Getter('currEditorType') currEditorType!: EditorType;
  @Getter('currEditorModel') currEditorModel!: EditorModel;
  @Getter('overviewEditor') overviewEditor!: EditorModel;
  @Getter('inCodeVault') inCodeVault!: boolean;
  @Mutation('switchEditor') switch!: (arg0: { editorType: EditorType; index: number }) => void;
  @Mutation('updateNodeInOverview') readonly updateNodeInOverview!: (cEditor: EditorModel) => void;

  private switchOverviewEditor() {
    // Save currEditorModel before switching as periodic save may not have captured last changes
    // and update overview editor if required
    this.updateNodeInOverview(this.currEditorModel);

    const oldEditorSaved: EditorSave = saveEditor(this.currEditorModel);
    const overviewEditorSave: EditorSave = saveEditor(this.overviewEditor);
    localStorage.setItem(`unsaved-editor-${this.currEditorModel.name}`, JSON.stringify(oldEditorSaved));
    localStorage.setItem('unsaved-editor-Overview', JSON.stringify(overviewEditorSave));

    this.switch({ editorType: EditorType.OVERVIEW, index: 0 });
  }

  private isSelected(editorType: EditorType) {
    return !this.inCodeVault && (this.currEditorType === editorType) ? 'selected' : '';
  }

  private displayNavbarContextualMenu(editorType: EditorType) {
    switch (editorType) {
      case EditorType.MODEL:
        this.isModelContextualMenuOpen = true;
        break;
      case EditorType.DATA:
        this.isDataContextualMenuOpen = true;
        break;
      default:
        break;
    }
  }

  private hideNavbarContextualMenu(): void {
    this.isModelContextualMenuOpen = false;
    this.isDataContextualMenuOpen = false;
  }
}

</script>

<style scoped>
  .left {
    background: var(--background);
    color: var(--foreground);
    border-right: 0.08rem solid var(--grey);
  }

  .tab-button {
    padding: 0.75rem 0;
    transition-duration: 0.1s;
    border-left-style: solid;
    border-left-width: 1px;
    border-left-color: var(--background);
  }

  .tab-button:hover {
    background: #1c1c1c;
    transition-duration: 0.1s;
    border-left-color: var(--blue);
    cursor: pointer;
  }

  .tab-button.selected {
    border-left-width: 4px;
    border-left-color: var(--blue);
  }

  .tab-icon {
    font-size: 1.5rem;
    height: 1.5rem;
  }

  hr {
    border-top: 0.1rem solid var(--dark-grey) !important;
  }

  .build {
    margin-top: 1rem;
  }

  .navbar-contextual-menu {
    position: absolute;
    margin-top: -35px;
    left: 59px;
    background: var(--background);
    color: var(--foreground);
    z-index: 1;
  }
</style>
