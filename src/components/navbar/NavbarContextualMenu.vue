<template>
  <div id="contextual-menu">
    <div v-for="(editor, index) in editors" :key="index">
      <div>
        <div id="row">
          <VerticalMenuButton
            :label="editor.name"
            :onClick="() => switchEditor({editorType, index})"
            :isSelected="editorType === currEditorType && index === currEditorIndex">
          </VerticalMenuButton>
          <div class="buttons">
            <RenameEditorButton :editorType="editorType" :index="index"/>
            <DeleteEditorButton :editorType="editorType" :index="index"/>
          </div>
        </div>
      </div>
    </div>
    <VerticalMenuButton
      :label="'+'"
      :onClick="this.createNewEditor"
      :isSelected="false"
    />
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import VerticalMenuButton from '@/components/buttons/VerticalMenuButton.vue';
import { mapGetters, mapMutations } from 'vuex';
import EditorType from '@/EditorType';
import { Getter, Mutation } from 'vuex-class';
import { EditorModel } from '@/store/editors/types';
import { uniqueTextInput } from '@/inputs/prompt';
import { saveEditor, SaveWithNames } from '@/file/EditorAsJson';
import RenameEditorButton from '@/components/buttons/RenameEditorButton.vue';
import DeleteEditorButton from '@/components/buttons/DeleteEditorButton.vue';

@Component({
  components: { VerticalMenuButton, RenameEditorButton, DeleteEditorButton },
  computed: mapGetters([
    'currEditorType',
    'currEditorIndex',
  ]),
  methods: mapMutations(['switchEditor']),
})
export default class NavbarContextualMenu extends Vue {
  @Prop({ required: true }) readonly editors!: EditorModel[];
  @Prop({ required: true }) readonly editorType!: EditorType;
  @Getter('editorNames') editorNames!: Set<string>;
  @Getter('saveWithNames') saveWithNames!: SaveWithNames;
  @Getter('currEditorModel') currEditorModel!: EditorModel;
  @Mutation('newEditor') newEditor!: (arg0: { editorType: EditorType; name: string }) => void;

  private createNewEditor(): void {
    const name: string | null = uniqueTextInput(this.editorNames,
      'Please enter a unique name for the editor');
    if (name !== null) {
      this.newEditor({ editorType: this.editorType, name });
      // Auto-Saving this new Editor
      this.$cookies.set('unsaved-project', this.saveWithNames);
      this.$cookies.set(`unsaved-editor-${name}`, saveEditor(this.currEditorModel));
    }
  }
}

</script>

<style lang="scss" scoped>
  #row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    &:hover {
      background: #1c1c1c;
      transition-duration: 0.1s;
      border-left-color: var(--blue);
      cursor: pointer;
    }
  }

  .buttons {
    display: flex;
  }

  #contextual-menu {
    border: 1px solid var(--grey);
  }
</style>
